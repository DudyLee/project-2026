"""Prepare and index the existing model's room and hourly inputs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

UI_ROOT = Path(__file__).resolve().parents[1]
DATASET = UI_ROOT / "dataset"
BIM_FILE = "SEND_BIM_476Rooms.csv"
SOURCE_FILE = "SEND_ML_BIM_EPW_07-18.csv"
WEATHER_FILE = "weather_hourly.csv"
TIME_FIELDS = ("DOY_1based", "Hour")
WEATHER_FIELDS = (
    "EPW_GHI_Wh_m2", "EPW_DNI_Wh_m2", "EPW_DHI_Wh_m2",
    "Solar_Altitude_mid_deg", "Solar_Azimuth_mid_deg",
)
ROOM_FIELDS = (
    "Room_Area_m2", "Glass_Area_m2", "Orientation_X_model", "Orientation_Y_model",
    "Solar_Obstruction_Ratio", "Annual_Incident_Shaded_kWh_m2",
)
SUPPORTED_TIMES = {(day, hour) for day in range(1, 366) for hour in range(7, 19)}


def records(path: Path, fields: tuple[str, ...]):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing = set(fields) - set(header)
        if missing or len(header) != len(set(header)):
            raise ValueError(f"{path.name}: invalid header, missing {sorted(missing)}")
        yield from reader


def validate_times(times: set[tuple[int, int]]) -> None:
    if times != SUPPORTED_TIMES:
        missing, extra = SUPPORTED_TIMES - times, times - SUPPORTED_TIMES
        raise ValueError(f"Invalid weather coverage: {len(missing)} missing, {len(extra)} extra")


def weather_values(row: dict) -> tuple[float, ...]:
    values = tuple(float(row[name]) for name in WEATHER_FIELDS)
    ghi, dni, dhi, altitude, azimuth = values
    if (not all(map(math.isfinite, values)) or min(ghi, dni, dhi) < 0
            or not -90 <= altitude <= 90 or not 0 <= azimuth <= 360):
        raise ValueError(f"Invalid weather values at day {row['DOY_1based']}, hour {row['Hour']}")
    return values


def build_weather(directory: Path = DATASET) -> int:
    unique: dict[tuple[int, int], tuple[float, ...]] = {}
    fields = TIME_FIELDS + WEATHER_FIELDS + ("Month", "Day", "HOY_0based")
    for row in records(directory / SOURCE_FILE, fields):
        day, hour = (int(row[name]) for name in TIME_FIELDS)
        expected_day = date(2001, int(row["Month"]), int(row["Day"])).timetuple().tm_yday
        if day != expected_day or int(row["HOY_0based"]) != (day - 1) * 24 + hour:
            raise ValueError(f"Inconsistent time index at day {day}, hour {hour}")
        values = weather_values(row)
        if unique.setdefault((day, hour), values) != values:
            raise ValueError(f"Conflicting weather at day {day}, hour {hour}")
    validate_times(set(unique))
    with NamedTemporaryFile(mode="w", dir=directory, encoding="utf-8", newline="", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.writer(stream)
            writer.writerow(TIME_FIELDS + WEATHER_FIELDS)
            for key, values in sorted(unique.items()):
                writer.writerow((*key, *values))
            stream.close()
            temporary.replace(directory / WEATHER_FILE)
        finally:
            temporary.unlink(missing_ok=True)
    return len(unique)


@dataclass(frozen=True)
class Room:
    id: str
    level: str
    area: float
    simulated: bool
    floor: int | None
    features: dict[str, float]

    def card(self) -> dict:
        return {"id": self.id, "level": self.level, "area": self.area, "simulated": self.simulated}


def floor_of(level: str) -> int | None:
    token = level.upper().split()
    if not token:
        return None
    if token[0] in {"G", "GF"}:
        return 0
    if token[0].endswith("F") and token[0][:-1].isdigit():
        return int(token[0][:-1])
    return None


class Catalog:
    def __init__(self, directory: Path = DATASET):
        self.rooms: dict[str, Room] = {}
        self.weather: dict[tuple[int, int], dict[str, float]] = {}
        for row in records(directory / BIM_FILE, ("Room_ID", "Level", "Sample_Role") + ROOM_FIELDS):
            room_id = row["Room_ID"].strip().upper()
            if not room_id or room_id in self.rooms:
                raise ValueError(f"Invalid or duplicate room ID: {room_id!r}")
            features = {name: float(row[name]) for name in ROOM_FIELDS}
            area, glass, x, y, obstruction, annual = features.values()
            if (not all(map(math.isfinite, features.values())) or min(area, glass, annual) <= 0
                    or not math.isclose(math.hypot(x, y), 1, abs_tol=1e-4, rel_tol=0)
                    or not 0 <= obstruction <= 1):
                raise ValueError(f"Room {room_id} has invalid model inputs")
            level = row["Level"].strip()
            self.rooms[room_id] = Room(room_id, level, area, row["Sample_Role"] in {"TRAIN", "TEST"},
                                       floor_of(level), features)
        for row in records(directory / WEATHER_FILE, TIME_FIELDS + WEATHER_FIELDS):
            key = tuple(int(row[name]) for name in TIME_FIELDS)
            if key in self.weather:
                raise ValueError(f"Duplicate weather key: {key}")
            self.weather[key] = dict(zip(WEATHER_FIELDS, weather_values(row)))
        validate_times(set(self.weather))

    def search(self, query: str = "", floor: int | None = None) -> list[Room]:
        query = query.strip().upper()
        rooms = (room for room in self.rooms.values()
                 if (floor is None or room.floor == floor)
                 and (not query or query in room.id or query in room.level.upper()))
        return sorted(rooms, key=lambda room: (room.id != query, not room.simulated, room.id))

    def inputs(self, room_id: str, day: int, hour: int) -> dict:
        room = self.rooms[room_id]
        return {
            "Room_ID": room_id,
            "DOY_1based": day,
            "Hour": hour,
            **room.features,
            **self.weather[day, hour],
        }


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET
    print(f"Prepared {build_weather(directory)} hourly records in {directory / WEATHER_FILE}")
