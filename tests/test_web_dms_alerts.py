"""Web DMS alert rule tests."""

import unittest
from unittest.mock import patch

from models.data_models import EyeResult, MouthResult, PoseResult


class _FakeFaceDetector:
    pass


with patch("detectors.face_detector.FaceDetector", _FakeFaceDetector):
    import web_app

from web_app import WebDetectionSystem


def _eye():
    return EyeResult(ear=0.3, is_closed=False, is_fatigued=False, frame_count=0)


def _mouth():
    return MouthResult(mar=0.2, is_yawning=False, is_fatigued=False, frame_count=0)


def _pose(yaw=0.0):
    return PoseResult(
        pitch=0.0,
        yaw=yaw,
        roll=0.0,
        is_head_down=False,
        is_fatigued=False,
        frame_count=0,
    )


def _system():
    system = object.__new__(WebDetectionSystem)
    system._dms_thresholds = {
        "eye_l1": 0.8,
        "eye_l2": 2.0,
        "head_down": 0.8,
        "yawn": 0.8,
        "look_away": 0.5,
        "phone": 0.1,
        "phone_use": 0.1,
        "smoke": 0.1,
        "no_driver": 1.0,
        "occlusion": 0.6,
        "yaw_abs": 15.0,
        "drinking": 0.1,
        "phone_eye_dist": 150.0,
        "cig_mouth_dist": 200.0,
        "drink_mouth_dist": 200.0,
    }
    system._dms_object_hold_sec = 1.0
    system._dms_object_seen_at = {
        "phone": 0.0,
        "smoke": 0.0,
        "cup": 0.0,
    }
    system._occlusion_cfg = {
        "dark_mean": 35.0,
        "low_var_std": 12.0,
    }
    system._dms_state = {
        "eye_closed_since": None,
        "head_down_since": None,
        "yawn_since": None,
        "look_away_since": None,
        "phone_since": None,
        "phone_use_since": None,
        "smoke_since": None,
        "no_driver_since": None,
        "occlusion_since": None,
        "drinking_since": None,
        "last_alert_signature": "",
    }
    system._add_log = lambda *args, **kwargs: None
    return system


def _set_centers(phone=(0.0, 0.0), cup=(0.0, 0.0), eyes=(0.0, 0.0), mouth=(0.0, 0.0)):
    web_app.PHONE_CENTER = phone
    web_app.CUP_CENTER = cup
    web_app.CIGARETTE_CENTER = (0.0, 0.0)
    web_app.EYES_CENTER = eyes
    web_app.MOUTH_CENTER = mouth


class WebDmsAlertTests(unittest.TestCase):
    def test_mobile_phone_label_triggers_phone_use_alert(self):
        now = 100.0
        system = _system()
        system._dms_state["phone_use_since"] = now - 1.0
        _set_centers(phone=(320.0, 320.0), eyes=(20.0, 20.0), mouth=(40.0, 40.0))

        with patch.object(web_app.time, "time", return_value=now):
            alerts = system._build_dms_alerts(["Mobile Phone"], _eye(), _mouth(), _pose())

        self.assertIn("玩手机预警", alerts)

    def test_near_phone_triggers_phone_call_alert(self):
        now = 100.0
        system = _system()
        system._dms_state["phone_since"] = now - 1.0
        _set_centers(phone=(45.0, 45.0), eyes=(40.0, 40.0), mouth=(50.0, 50.0))

        with patch.object(web_app.time, "time", return_value=now):
            alerts = system._build_dms_alerts(["Cell Phone"], _eye(), _mouth(), _pose())

        self.assertIn("打电话预警", alerts)

    def test_water_cup_near_mouth_triggers_drinking_alert(self):
        now = 100.0
        system = _system()
        system._dms_state["drinking_since"] = now - 1.0
        _set_centers(cup=(110.0, 110.0), mouth=(100.0, 100.0))

        with patch.object(web_app.time, "time", return_value=now):
            alerts = system._build_dms_alerts(["Water Cup"], _eye(), _mouth(), _pose())

        self.assertIn("喝水预警", alerts)

    def test_cigarette_label_triggers_smoking_alert(self):
        now = 100.0
        system = _system()
        system._dms_state["smoke_since"] = now - 1.0
        _set_centers()

        with patch.object(web_app.time, "time", return_value=now):
            alerts = system._build_dms_alerts(["Cigarettes"], _eye(), _mouth(), _pose())

        self.assertIn("抽烟预警", alerts)

    def test_recent_phone_detection_is_held_through_short_yolo_miss(self):
        system = _system()
        system._dms_object_seen_at["phone"] = 99.4
        system._dms_state["phone_use_since"] = 99.0
        _set_centers(phone=(0.0, 0.0))

        with patch.object(web_app.time, "time", return_value=100.0):
            alerts = system._build_dms_alerts([], _eye(), _mouth(), _pose())

        self.assertIn("玩手机预警", alerts)

    def test_look_away_uses_yaw_rule_not_yolo_label(self):
        now = 100.0
        system = _system()
        system._dms_state["look_away_since"] = now - 1.0
        _set_centers()

        with patch.object(web_app.time, "time", return_value=now):
            alerts = system._build_dms_alerts(["look away"], _eye(), _mouth(), _pose(yaw=0.0))

        self.assertTrue(all("左顾右盼" not in alert for alert in alerts))

    def test_large_yaw_triggers_look_away_alert(self):
        now = 100.0
        system = _system()
        system._dms_state["look_away_since"] = now - 1.0
        _set_centers()

        with patch.object(web_app.time, "time", return_value=now):
            alerts = system._build_dms_alerts([], _eye(), _mouth(), _pose(yaw=16.0))

        self.assertIn("左顾右盼预警（视线偏移≥0.5秒）", alerts)


if __name__ == "__main__":
    unittest.main()
