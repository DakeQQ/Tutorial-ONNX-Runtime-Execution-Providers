#!/usr/bin/env python3
"""Backward-compatible launcher for the canonical AMD provider test."""

from provider_test import main


if __name__ == "__main__":
    raise SystemExit(main())