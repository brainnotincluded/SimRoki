from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from robot_sim.client import SimulatorClient


JOINTS = (
    ("right_hip", -1.6, 1.6, -0.15),
    ("right_knee", 0.0, 2.2, 1.15),
    ("left_hip", -1.6, 1.6, -0.15),
    ("left_knee", 0.0, 2.2, 1.15),
)


class ServoSliderApp:
    def __init__(self) -> None:
        self.client = SimulatorClient()
        self.root = tk.Tk()
        self.root.title("Robot Servo Sliders")
        self.root.geometry("520x340")
        self.root.configure(bg="#f5f1ea")

        self.status_var = tk.StringVar(value="ready")
        self.scale_vars: dict[str, tk.DoubleVar] = {}
        self.value_vars: dict[str, tk.StringVar] = {}
        self.after_id: str | None = None

        self._build_ui()
        self._load_initial_state()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Five-Link Servo Control")
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        for idx, (joint, min_value, max_value, default) in enumerate(JOINTS, start=1):
            ttk.Label(frame, text=joint).grid(row=idx, column=0, sticky="w", padx=(0, 10), pady=8)

            scale_var = tk.DoubleVar(value=default)
            value_var = tk.StringVar(value=f"{default:.3f} rad")
            self.scale_vars[joint] = scale_var
            self.value_vars[joint] = value_var

            scale = tk.Scale(
                frame,
                from_=min_value,
                to=max_value,
                orient=tk.HORIZONTAL,
                resolution=0.01,
                length=260,
                variable=scale_var,
                command=lambda _value, j=joint: self._on_slider_change(j),
                bg="#f5f1ea",
                highlightthickness=0,
            )
            scale.grid(row=idx, column=1, sticky="ew", pady=4)

            ttk.Label(frame, textvariable=value_var, width=12).grid(row=idx, column=2, sticky="e")

        frame.columnconfigure(1, weight=1)

        button_row = ttk.Frame(frame)
        button_row.grid(row=len(JOINTS) + 1, column=0, columnspan=3, sticky="ew", pady=(14, 10))

        ttk.Button(button_row, text="Send Pose", command=self.send_all).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Reset Robot", command=self.reset_robot).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Pause", command=self.pause).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Resume", command=self.resume).pack(side=tk.LEFT)

        ttk.Label(frame, textvariable=self.status_var).grid(
            row=len(JOINTS) + 2, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

    def _load_initial_state(self) -> None:
        try:
            state = self.client.get_state()
            joints = state.get("joints", {})
            for joint, _min_value, _max_value, default in JOINTS:
                angle = joints.get(joint, {}).get("target", default)
                self.scale_vars[joint].set(angle)
                self.value_vars[joint].set(f"{angle:.3f} rad")
            self.status_var.set("connected to local simulator")
        except Exception as exc:
            self.status_var.set(f"cannot read state: {exc}")

    def _on_slider_change(self, joint: str) -> None:
        value = self.scale_vars[joint].get()
        self.value_vars[joint].set(f"{value:.3f} rad")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(40, self.send_all)

    def current_targets(self) -> dict[str, float]:
        return {joint: var.get() for joint, var in self.scale_vars.items()}

    def send_all(self) -> None:
        try:
            self.client.set_targets(self.current_targets())
            self.status_var.set("targets sent")
        except Exception as exc:
            self.status_var.set(f"send failed: {exc}")

    def reset_robot(self) -> None:
        try:
            self.client.reset()
            self._load_initial_state()
            self.status_var.set("robot reset")
        except Exception as exc:
            self.status_var.set(f"reset failed: {exc}")

    def pause(self) -> None:
        try:
            self.client.pause()
            self.status_var.set("paused")
        except Exception as exc:
            self.status_var.set(f"pause failed: {exc}")

    def resume(self) -> None:
        try:
            self.client.resume()
            self.status_var.set("running")
        except Exception as exc:
            self.status_var.set(f"resume failed: {exc}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ServoSliderApp().run()
