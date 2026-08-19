"""Minimal quaternion helper class for orientation integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_array(values: np.ndarray) -> "Quaternion":
        return Quaternion(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle_rad: float) -> "Quaternion":
        norm = np.linalg.norm(axis)
        if norm <= 0.0 or abs(angle_rad) <= 0.0:
            return Quaternion.identity()
        direction = axis / norm
        half_angle = 0.5 * angle_rad
        sin_half = np.sin(half_angle)
        return Quaternion(np.cos(half_angle), *(sin_half * direction)).normalized()

    @staticmethod
    def from_rotation_vector(rotation_vector_rad: np.ndarray) -> "Quaternion":
        """Exponential-map rotation quaternion for a rotation vector: axis is
        the vector direction, angle is its length. Below the 1e-6 length
        threshold, returns the identity quaternion rather than dividing by a
        near-zero length.
        """
        length = float(np.linalg.norm(rotation_vector_rad))
        if length < 0.000001:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        sin_half = np.sin(length / 2.0)
        cos_half = np.cos(length / 2.0)
        return Quaternion(
            cos_half,
            sin_half * rotation_vector_rad[0] / length,
            sin_half * rotation_vector_rad[1] / length,
            sin_half * rotation_vector_rad[2] / length,
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    def normalized(self) -> "Quaternion":
        values = self.as_array()
        norm = np.linalg.norm(values)
        if norm <= 0.0:
            return Quaternion.identity()
        return Quaternion.from_array(values / norm)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other: "Quaternion") -> "Quaternion":
        w1, x1, y1, z1 = self.as_array()
        w2, x2, y2, z2 = other.as_array()
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def rotate_vector(self, vector: np.ndarray) -> np.ndarray:
        pure = Quaternion(0.0, float(vector[0]), float(vector[1]), float(vector[2]))
        rotated = self.multiply(pure).multiply(self.conjugate())
        return np.array([rotated.x, rotated.y, rotated.z], dtype=float)