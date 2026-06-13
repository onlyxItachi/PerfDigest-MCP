"""Platform layer — the Metal / PowerShell / Linux / WSL split.

``detect`` identifies the host (OS, WSL, shell, GPU vendors, profilers on PATH);
``shell`` builds platform-correct command strings (POSIX vs PowerShell quoting,
WSL path translation); ``capabilities`` gates which backends can *capture* here.
Reading/digesting a report is never gated by this layer.
"""
