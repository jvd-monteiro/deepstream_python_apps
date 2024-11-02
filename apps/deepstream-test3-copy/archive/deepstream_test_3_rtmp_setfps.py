#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../../')
from pathlib import Path
from os import environ
import gi
import configparser
import argparse
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA

import pyds

no_display = False
file_loop = False
perf_data = None
measure_latency = False

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_OK = 0
PGIE_CLASS_ID_SCAPCOL = 1
PGIE_CLASS_ID_SCAP = 2
PGIE_CLASS_ID_SCOL = 3
PGIE_CLASS_ID_EMP = 4
PGIE_CLASS_ID_CAR = 5
PGIE_CLASS_ID_CMN = 6
PGIE_CLASS_ID_IDK = 7
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1
pgie_classes_str = ["OK", "S/CapCol", "S/Cap", "S/Col", "EMP", "CAR", "CMN", "IDK"]

import csv
from datetime import datetime
from collections import deque

# Constants
ALERT_THRESHOLD = 100  # Minimum number of frames with alerts in the last 150
FRAME_BUFFER_SIZE = 150  # The number of frames to store per camera

# Buffers to store alert counts per camera
alert_buffers = {}

# CSV filename for logging
CSV_FILENAME = "../alerts_log.csv"


# IoU Calculation Function
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area_box1 + area_box2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


# pgie_src_pad_buffer_probe  will extract metadata received on tiler sink pad
# and update params for drawing rectangle, object information, etc.
# CSV filename for logging
CSV_FILENAME = "../alerts_log.csv"

# Function to log the camera, timestamp, and average confidence when the alert condition is met
def log_alert(camera_index, average_confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([camera_index, timestamp, f"Avg Confidence: {average_confidence:.2f}"])
    print(f"Logged alert for camera {camera_index} at {timestamp} with avg confidence {average_confidence:.2f}")


# pgie_src_pad_buffer_probe  will extract metadata received on tiler sink pad
# and update params for drawing rectangle, object information, etc.
def pgie_src_pad_buffer_probe(pad, info, u_data):
    global alert_buffers
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        source_index = frame_meta.pad_index
        if source_index not in alert_buffers:
            alert_buffers[source_index] = deque(maxlen=FRAME_BUFFER_SIZE)

        objects = []  # Store all valid detected objects here
        confidences = []  # Confidence list for alert-triggering problematic classes

        # Step 1: Iterate through objects and filter out problematic classes with confidence < 0.6
        l_obj = frame_meta.obj_meta_list
        prev_l_obj = None  # Keep track of the previous object to modify the linked list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            bbox = (
                obj_meta.rect_params.left,
                obj_meta.rect_params.top,
                obj_meta.rect_params.left + obj_meta.rect_params.width,
                obj_meta.rect_params.top + obj_meta.rect_params.height,
            )
            confidence = obj_meta.confidence

            # Discard problematic classes with confidence < 0.6
            if obj_meta.class_id in [PGIE_CLASS_ID_SCAPCOL, PGIE_CLASS_ID_SCAP, PGIE_CLASS_ID_SCOL] and confidence < 0.6:
                # Properly remove the object from the metadata
                next_l_obj = l_obj.next
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)  # Remove object metadata
                l_obj = next_l_obj
                continue  # Skip this detection

            # Keep confidence for alert analysis (for problematic cases only)
            if obj_meta.class_id in [PGIE_CLASS_ID_SCAPCOL, PGIE_CLASS_ID_SCAP, PGIE_CLASS_ID_SCOL]:
                confidences.append(confidence)

            # Add valid objects for NMS processing
            objects.append((bbox, confidence, obj_meta))
            prev_l_obj = l_obj
            l_obj = l_obj.next

        # If no valid objects are left, exit early
        if not objects:
            return Gst.PadProbeReturn.OK

        # Step 2: Apply NMS across all remaining valid detections with an IoU threshold of 0.4
        suppressed_objects = set()
        for i in range(len(objects)):
            bbox1, conf1, obj1 = objects[i]
            for j in range(i + 1, len(objects)):
                bbox2, conf2, obj2 = objects[j]
                iou = calculate_iou(bbox1, bbox2)

                if iou > 0.4:  # Apply NMS if IoU > 0.4
                    if conf1 > conf2:
                        suppressed_objects.add(obj2)
                    else:
                        suppressed_objects.add(obj1)

        # Step 3: Remove suppressed objects from metadata
        l_obj = frame_meta.obj_meta_list
        prev_l_obj = None  # Track the previous object in the linked list again

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Remove suppressed objects (bounding box + label)
            if obj_meta in suppressed_objects:
                next_l_obj = l_obj.next
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)  # Completely remove suppressed object
                l_obj = next_l_obj
                continue  # Move on to the next object

            prev_l_obj = l_obj
            l_obj = l_obj.next

        # Step 4: Display remaining objects with labels and confidence (for non-suppressed objects)
        for bbox, confidence, obj_meta in objects:
            if obj_meta in suppressed_objects:
                continue  # Skip suppressed objects

            # Set bounding box color based on class
            if obj_meta.class_id in [PGIE_CLASS_ID_SCAPCOL, PGIE_CLASS_ID_SCAP, PGIE_CLASS_ID_SCOL]:
                obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)  # Red for problematic classes
            else:
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0)  # Blue for other classes

            # Display class label and confidence
            obj_meta.text_params.display_text = f"{pgie_classes_str[obj_meta.class_id]} {round(confidence, 2)}"
            obj_meta.text_params.font_params.font_size = 8  # Consistent font size
            obj_meta.text_params.set_bg_clr = 1
            obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            obj_meta.text_params.y_offset = max(0, int(obj_meta.rect_params.top) - 25) # 25 pixels above bounding box

        # Step 5: Handle alert generation based on problematic objects
        total_alert_objects = len([obj for bbox, conf, obj in objects if obj.class_id in [PGIE_CLASS_ID_SCAPCOL, PGIE_CLASS_ID_SCAP, PGIE_CLASS_ID_SCOL]])

        # Store alert status in buffer
        alert_buffers[source_index].append(1 if total_alert_objects > 0 else 0)

        # Log alerts if threshold is met
        if sum(alert_buffers[source_index]) >= ALERT_THRESHOLD:
            if confidences:
                average_confidence = sum(confidences) / len(confidences)
            else:
                average_confidence = 0.0
            log_alert(source_index, average_confidence)
            alert_buffers[source_index].clear()

        # Update FPS performance data
        stream_index = f"stream{source_index}"
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    print("gstname=", gstname)
    if gstname.find("video") != -1:
        print("features=", features)
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") is not None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    print("Creating source bin")

    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")

    if file_loop:
        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin\n")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None
    return nbin


def main(args, requested_pgie=None, config=None, disable_probe=False):
    global perf_data
    perf_data = PERF_DATA(len(args))

    number_sources = len(args)

    platform_info = PlatformInfo()
    Gst.init(None)

    # Create pipeline
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
    print("Creating streammux\n")

    # Create the streammux element
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")

    pipeline.add(streammux)

    for i in range(number_sources):
        print(f"Creating source_bin {i}\n")
        uri_name = args[i]
        if uri_name.startswith("rtsp://"):
            is_live = True

        # Create source bin
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin\n")
        pipeline.add(source_bin)

        # Create a videorate element to control the frame rate
        videorate = Gst.ElementFactory.make("videorate", f"videorate_{i}")
        if not videorate:
            sys.stderr.write("Unable to create videorate\n")
        videorate.set_property("max-rate", 4)  # Set FPS

        pipeline.add(videorate)

        # Link source_bin to videorate
        srcpad = source_bin.get_static_pad("src")
        sinkpad = streammux.request_pad_simple(f"sink_{i}")
        if not srcpad or not sinkpad:
            sys.stderr.write("Unable to link source to streammux\n")
        srcpad.link(videorate.get_static_pad("sink"))
        videorate.get_static_pad("src").link(sinkpad)

    # Create the queue elements
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")

    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)

    # Initialize nvdslogger to None
    nvdslogger = None

    # Create Primary inference engine (Pgie)
    print("Creating Pgie\n")
    if requested_pgie == "nvinferserver" or requested_pgie == "nvinferserver-grpc":
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")

    if not pgie:
        sys.stderr.write(f"Unable to create pgie: {requested_pgie}\n")
    pipeline.add(pgie)

    # Check if logger should be used
    if disable_probe:
        print("Creating nvdslogger\n")
        nvdslogger = Gst.ElementFactory.make("nvdslogger", "nvdslogger")
        pipeline.add(nvdslogger)

    # Create tiler, video converter, and OSD
    print("Creating tiler\n")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write("Unable to create tiler\n")
    pipeline.add(tiler)

    print("Creating nvvidconv\n")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")
    pipeline.add(nvvidconv)

    print("Creating nvosd\n")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")
    nvosd.set_property("process-mode", OSD_PROCESS_MODE)
    nvosd.set_property("display-text", OSD_DISPLAY_TEXT)
    pipeline.add(nvosd)

    # Memory settings for file loop scenarios
    if file_loop:
        if platform_info.is_integrated_gpu():
            streammux.set_property("nvbuf-memory-type", 4)
        else:
            streammux.set_property("nvbuf-memory-type", 2)

    # Configure sink for output (fakesink or display sink)
    if no_display:
        print("Creating Fakesink\n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        sink.set_property("enable-last-sample", 0)
        sink.set_property("sync", 0)
    else:
        if platform_info.is_integrated_gpu():
            print("Creating nv3dsink\n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink\n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write("Unable to create sink element\n")
        pipeline.add(sink)

    if is_live:
        streammux.set_property("live-source", 1)

    # Set streammux properties
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    if requested_pgie in ["nvinferserver", "nvinferserver-grpc", "nvinfer"] and config is not None:
        pgie.set_property("config-file-path", config)
    else:
        pgie.set_property("config-file-path", "config_infer_primary_yoloV8.txt")

    # Override batch size if necessary
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            f"WARNING: Overriding infer-config batch-size {pgie_batch_size} with number of sources {number_sources}\n")
        pgie.set_property("batch-size", number_sources)

    # Set tiler properties
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil(1.0 * number_sources / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)

    # Link elements in the pipeline
    print("Linking elements in the Pipeline\n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    if nvdslogger:
        queue2.link(nvdslogger)
        nvdslogger.link(tiler)
    else:
        queue2.link(tiler)
    tiler.link(queue3)
    queue3.link(nvvidconv)
    nvvidconv.link(queue4)
    queue4.link(nvosd)
    nvosd.link(queue5)
    queue5.link(sink)

    # Set up the GStreamer main loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Attach probe to pgie src pad for metadata inspection
    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write("Unable to get src pad\n")
    else:
        if not disable_probe:
            pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
            GLib.timeout_add(10000, perf_data.perf_print_callback)

    if environ.get("NVDS_ENABLE_LATENCY_MEASUREMENT") == "1":
        print(
            "Pipeline Latency Measurement enabled!\nPlease set env var NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1 for Component Latency Measurement")
        global measure_latency
        measure_latency = True

    # Start playing the pipeline
    print("Now playing...")
    for i, source in enumerate(args):
        print(f"{i} : {source}")

    print("Starting pipeline\n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)



def parse_args():

    parser = argparse.ArgumentParser(
        prog="deepstream_test_3", description="deepstream-test3 multi stream, multi model inference reference app"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=["a"],
        required=True,
    )
    parser.add_argument(
        "-c",
        "--configfile",
        metavar="config_location.txt",
        default=None,
        help="Choose the config-file to be used with specified pgie",
    )
    parser.add_argument(
        "-g",
        "--pgie",
        default=None,
        help="Choose Primary GPU Inference Engine",
        choices=["nvinfer", "nvinferserver", "nvinferserver-grpc"],
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        dest="no_display",
        help="Disable display of video output",
    )
    parser.add_argument(
        "--file-loop",
        action="store_true",
        default=False,
        dest="file_loop",
        help="Loop the input file sources after EOS",
    )
    parser.add_argument(
        "--disable-probe",
        action="store_true",
        default=False,
        dest="disable_probe",
        help="Disable the probe function and use nvdslogger for FPS",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    stream_paths = args.input
    pgie = args.pgie
    config = args.configfile
    disable_probe = args.disable_probe
    global no_display
    global file_loop
    no_display = args.no_display
    file_loop = args.file_loop

    if config and not pgie or pgie and not config:
        sys.stderr.write("\nEither pgie or configfile is missing. Please specify both! Exiting...\n\n\n\n")
        parser.print_help()
        sys.exit(1)
    if config:
        config_path = Path(config)
        if not config_path.is_file():
            sys.stderr.write(f"Specified config-file: {config} doesn't exist. Exiting...\n\n")
            sys.exit(1)

    print(vars(args))
    return stream_paths, pgie, config, disable_probe


if __name__ == "__main__":
    stream_paths, pgie, config, disable_probe = parse_args()
    sys.exit(main(stream_paths, pgie, config, disable_probe))