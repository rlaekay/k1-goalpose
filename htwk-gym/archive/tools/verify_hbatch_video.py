#!/usr/bin/env python3
"""Verify that an HBatch video eval completed before native sim teardown."""

import argparse
import hashlib
import json
import os
import sys

import imageio


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def verify(directory, completion_token):
    marker_path = os.path.join(directory, "eval-complete-codex.json")
    report_path = os.path.join(directory, "report.json")
    video_path = os.path.join(directory, "rollout_env0.mp4")

    require(os.path.isfile(marker_path), "completion marker is missing")
    with open(marker_path, encoding="utf-8") as f:
        marker = json.load(f)
    require(marker.get("version") == 1 and marker.get("status") == "complete",
            "completion marker has the wrong schema or status")
    require(marker.get("completion_token") == completion_token,
            "completion marker belongs to a different eval invocation")

    artifacts = marker.get("artifacts") or {}
    for name, path in (("report.json", report_path),
                       ("rollout_env0.mp4", video_path)):
        expected = artifacts.get(name) or {}
        require(os.path.isfile(path), "{} is missing".format(name))
        size = os.path.getsize(path)
        require(size > 0, "{} is empty".format(name))
        require(size == int(expected.get("bytes", -1)),
                "{} size differs from completion marker".format(name))
        require(sha256_file(path) == expected.get("sha256"),
                "{} hash differs from completion marker".format(name))

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    disturbance = report.get("disturbance_eval") or {}
    counters = {
        "events": int(disturbance.get("events", 0)),
        "recorded": int(disturbance.get("video_recorded_frames", 0)),
        "force_active": int(disturbance.get("video_force_active_frames", 0)),
        "red_arrow": int(disturbance.get("video_force_arrow_drawn_frames", 0)),
        "path_carrot": int(disturbance.get("video_path_carrot_drawn_frames", 0)),
        "path_trace": int(disturbance.get("video_path_trace_drawn_frames", 0)),
    }
    require(all(value > 0 for value in counters.values()),
            "required video/event counter is zero: {}".format(counters))
    require(all(counters[name] <= counters["recorded"] for name in
                ("force_active", "red_arrow", "path_carrot", "path_trace")),
            "frame counter exceeds recorded frame count: {}".format(counters))

    reader = imageio.get_reader(video_path)
    decoded_frames = 0
    try:
        for frame in reader:
            require(getattr(frame, "size", 0) > 0,
                    "decoded an empty video frame")
            decoded_frames += 1
    finally:
        reader.close()
    require(decoded_frames == counters["recorded"],
            "decoded {} frames but report attests {}".format(
                decoded_frames, counters["recorded"]))
    return counters, decoded_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--completion_token", required=True)
    args = parser.parse_args()
    try:
        counters, decoded = verify(
            os.path.abspath(args.directory), args.completion_token)
    except Exception as exc:
        print("FAIL  HBatch video artifact verification: {}".format(exc),
              flush=True)
        return 1
    print("PASS  HBatch video artifacts: {} decoded frames; {}".format(
        decoded, counters), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
