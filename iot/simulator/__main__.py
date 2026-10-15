"""PowerPredict MQTT simulator CLI. Publishes SIMULATED readings only."""

from __future__ import annotations

import argparse
import logging
import os

from iot.env import mqtt_settings
from iot.simulator.buildings import resolve_building_ids
from iot.simulator.patterns import generate_reading
from iot.simulator.payload import MQTT_ORIGINS, SCHEMA_VERSION, topic_filter, topic_for
from iot.simulator.publisher import SIMULATED_BANNER, run_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m iot.simulator",
        description=(
            "Publish SIMULATED building sensor readings to Mosquitto. "
            "data_origin is always simulated_iot. This is not live physical sensor data. "
            "Historical CSV ingest never uses this publisher."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one simulated payload per building and exit (no MQTT connect).",
    )
    parser.add_argument(
        "--buildings",
        nargs="+",
        metavar="ID",
        help="Building ids to publish. Default: pipeline selection, else building_001/002/003.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Wall-clock seconds between publish ticks (default: 5).",
    )
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Faster demo tick (1s interval unless --interval is already below 5s).",
    )
    parser.add_argument(
        "--tick-minutes",
        type=float,
        default=0.0,
        help="If > 0, advance a simulated clock by this many minutes each tick.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Publish this many ticks then exit (default: run until Ctrl+C).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Broker host (default: MQTT_HOST from .env / environment).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Broker port (default: MQTT_PORT from .env / environment).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s SIMULATED %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = mqtt_settings()
    if args.host:
        settings["host"] = args.host
    if args.port:
        settings["port"] = args.port
    buildings = resolve_building_ids(args.buildings)
    root = settings["topic_root"]

    print(SIMULATED_BANNER.strip())
    print(f"Allowed MQTT data_origin values: {', '.join(sorted(MQTT_ORIGINS))}")
    print(f"schema_version={SCHEMA_VERSION}  occupancy_source=simulated")
    print(f"Subscribe filter: {topic_filter(root)}")
    print(f"Buildings: {', '.join(buildings)}")
    print("Historical CSV ingest must not use this publisher.")
    if os.environ.get("MQTT_PASSWORD"):
        print(f"Broker: {settings['host']}:{settings['port']} (credentials from env, not source)")
    else:
        print(f"Broker: {settings['host']}:{settings['port']} (set MQTT_USERNAME / MQTT_PASSWORD in .env)")

    if args.dry_run:
        for building_id in buildings:
            payload = generate_reading(building_id)
            print(f"{topic_for(building_id, root=root)}  {payload}")
        return 0

    return run_loop(
        settings=settings,
        building_ids=buildings,
        interval_s=args.interval,
        ticks=args.ticks,
        accelerate=args.accelerate,
        tick_minutes=args.tick_minutes,
        client_id=f"powerpredict-simulator-{os.getpid()}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
