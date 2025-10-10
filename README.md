# LLuMa Musume

Experimental harness for Uma Musume gameplay on Windows by an LLM agent.

## Quickstart (capture layer)

1. Install dependencies with `pip install -r requirements.txt` on Windows 11.
2. Populate `config.yaml` (see `AGENTS.md` for recommended values). Defaults fall back to a 1920x1080 window at the top-left of the primary display with 150% scaling.
3. Run a one-off capture via:

   ```bash
   python -m lluma_os.cli --reposition --log-level DEBUG
   ```

   Captures are stored under `captures/` as PNG files (`<turn_id>.png`, plus `-primary` and `-menus` crops when enabled).
