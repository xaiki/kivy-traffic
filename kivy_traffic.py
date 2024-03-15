"""
Kivy asyncio example app.
Kivy needs to run on the main thread and its graphical instructions have to be
called from there.  But it's still possible to run an asyncio EventLoop, it
just has to happen on its own, separate thread.
Requires Python 3.5+.
"""

import kivy

kivy.require('1.10.0')

import asyncio
import threading

from kivy.app import App
from kivy.clock import mainthread, Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.properties import Property, ListProperty, ObjectProperty, BooleanProperty, NumericProperty, ColorProperty

from random import random
import argparse
import cv2
from tqdm import tqdm

from video import VideoProcessor, COLORS, COLOR_FMT

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def to_int(lst):
    for i in lst:
        yield int(i)

def get_color(id):
    c = COLORS.colors[id]
    return [c.r/255.0, c.g/255.0, c.b/255.0, 1.0]


class PaintWidget(Widget):
    zid = NumericProperty(0)
    color = ColorProperty(None)
    current = ListProperty([])
    zone = Property(None)
    line = None
    cur_line = None
    def __init__(self, zid, **kwargs):
        super().__init__(**kwargs)
        self.zid = zid
        self.color = get_color(zid)

    def on_touch_down(self, touch):
        if len(self.current) > 1:
            return

        with self.canvas:
            Color(*self.color)
            d = 30.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            if not self.line:
                self.line = Line()
                self.cur_line = []
            self.line.points += [touch.x, touch.y]
            self.cur_line.append([touch.x/self.width, 1.0 - touch.y/self.height])
            if len(self.line.points) == 8:
                self.line.points += [self.line.points[0], self.line.points[1]]
                self.cur_line.append(self.cur_line[0])
                self.current.append(self.cur_line)
                if len(self.current) == 2:
                    self.zone = self.current
                    # print(self.current[0])
                    # print(to_int(self.current[0]))
                    # print(list(to_int(self.current[0])))
                    # self.zone = [
                    #     list(chunks(list(to_int(self.current[0])), 2)),
                    #     list(chunks(list(to_int(self.current[1])), 2))
                    # ]
                self.line = Line()
                self.cur_line = []
            print(self.line.points)


class ZoneShow(Widget):
    zones = ListProperty([])
    def on_zones(self, instance, zones):
        print(instance, zones)
        self.canvas.clear()

        for i, zone in enumerate(zones):
            with self.canvas:
                Color(*get_color(i + 1))
                Line(points=zone[0])
                Line(points=zone[1])

class VideoWidget(BoxLayout):
    texture = Property(Texture.create(size=(640,640)))

    text = ObjectProperty(Label(text="drag a video"))
    loaded = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(self.text)


    def load_texture(self, frame):
        print("loading frame")
        buf = cv2.flip(frame, 0).tobytes()
        size = (frame.shape[1], frame.shape[0])
        texture = Texture.create(size=size, colorfmt=COLOR_FMT)
        texture.blit_buffer(buf, colorfmt=COLOR_FMT, bufferfmt='ubyte')
        self.texture = texture

class RootLayout(BoxLayout):
    pass

class EventLoopWorker(EventDispatcher):
    __events__ = ('on_frame',)  # defines this EventDispatcher's sole event

    def __init__(self, **kwargs):
        super().__init__()
        self._thread = threading.Thread(target=self._run_loop)  # note the Thread target here
        self._thread.daemon = True
        self.loop = None
        self.source_path = None
        self._frame_task = None
        self._video_proc = VideoProcessor(**kwargs)
        self._frame_generator = None
        self._video_info = None
        self._current_frame = None

    def _run_loop(self):
        self.loop = asyncio.get_event_loop_policy().new_event_loop()
        asyncio.set_event_loop(self.loop)
        # this example doesn't include any cleanup code, see the docs on how
        # to properly set up and tear down an asyncio event loop
        self.loop.run_forever()

    def start(self):
        self._thread.start()

    async def process_frame(self):
        print("start frame", self._video_proc)
        @mainthread
        def notify(frame):
            self.dispatch('on_frame', frame)

        for frame in tqdm(self._video_proc.frames_generator,
                          total = self._video_proc.video_info.total_frames):
            annotated_frame = self._video_proc.process_frame(frame)
            notify(annotated_frame)
            #await asyncio.sleep(1.0 / 60.0)
        print("end frame")

    async def frame(self):
        @mainthread
        def notify(frame):
            self.dispatch('on_frame', frame)
        try:
            print("try frame")
            if self._current_frame is None:
                self._current_frame = next(self._video_proc.frames_generator)
            print("try annotate")
            annotated_frame = self._video_proc.draw_zones(self._current_frame)
            notify(annotated_frame)
        except Exception as e:
            print("frame ERROR", e)
            raise e

    def set_source_path(self, source_path):
        self.source_path = source_path
        self._video_proc.load_video(source_path)
        self._current_frame = None
        return self._restart_frame()

    def _restart_frame(self):
        """Helper to start/reset the frames task when a video is loaded"""
        if self._frame_task is not None:
            self._frame_task.cancel()
        self._frame_task = self.loop.create_task(self.frame())

    def unnormalize_zone(self, zone):
        print("normalize", zone)
        w, h = self._video_proc.video_info.resolution_wh

        return [[int(x*w), int(y*h)] for x, y in zone]

    def set_zones(self, zones):
        proc_zones = {"in": [], "out": []}
        for zone in zones:
            proc_zones["in"].append(self.unnormalize_zone(zone[0]))
            proc_zones["out"].append(self.unnormalize_zone(zone[1]))

        self._video_proc.zones = proc_zones
        # HACK(xaiki): it's done to reload the zones
        self._video_proc.load_video(self.source_path)
        self._restart_frame()

    def start_process(self):
        """Helper to start/reset the frames task when a video is loaded"""
        if self._frame_task is not None:
            self._frame_task.cancel()
        self._frame_task = self.loop.create_task(self.process_frame())

    def on_frame(self, *_):
        """An EventDispatcher event must have a corresponding method."""
        pass


class TrafficAnalysisApp(App):
    painter: Widget

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.event_loop_worker = None
        self.zones = []

    def build(self):
        layout = RootLayout()
        Window.bind(on_drop_file=self._on_drop_file)
        Clock.schedule_once(lambda dt: self.start_event_loop_thread())
        return layout

    def _on_drop_file(self, window, file_path, x, y, *args):
        worker = self.event_loop_worker
        if worker is not None:
            loop = self.event_loop_worker.loop
            # use the thread safe variant to run it on the asyncio event loop:
            loop.call_soon_threadsafe(worker.set_source_path, file_path.decode('utf-8'))

    def start_event_loop_thread(self):
        """Start the asyncio event loop thread. Bound to the top button."""
        if self.event_loop_worker is not None:
            return
        self.event_loop_worker = worker =  EventLoopWorker(**self.kwargs)

        def display_on_frame(instance, frame):
            self.root.ids.video.load_texture(frame)

        # make the label react to the worker's `on_pulse` event:
        worker.bind(on_frame=display_on_frame)
        worker.start()

    def add_zone(self):
        self.painter = painter = PaintWidget(len(self.zones))
        self.root.ids.anchor.add_widget(painter)
        def on_zone(instance, zone):
            self.zones.append(zone)
            #self.root.ids.zones.zones = self.zones
            self.root.ids.add_zone_btn.state = 'normal'
            self.sync_zones(self.zones)
            self.root.ids.anchor.remove_widget(painter)
        painter.bind(zone=on_zone)

        print("add zone")

    def remove_zone(self):
        self.zones.pop()
        self.root.ids.zones.zones = self.zones

    def sync_zones(self, zones):
        worker = self.event_loop_worker
        if worker is not None:
            loop = self.event_loop_worker.loop
            # use the thread safe variant to run it on the asyncio event loop:
            loop.call_soon_threadsafe(worker.set_zones, self.zones)

    def start_process(self):
        worker = self.event_loop_worker
        if worker is not None:
            loop = self.event_loop_worker.loop
            # use the thread safe variant to run it on the asyncio event loop:
            loop.call_soon_threadsafe(worker.start_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        default=None,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    TrafficAnalysisApp(
        source_weights_path=args.source_weights_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    ).run()
