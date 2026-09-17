# 🌐 Getting a PUBLIC link for the RailPulse dashboard

You have 3 options — from fastest (temporary) to permanent.

---

## Option A — One-command public link (built-in, no installs)

Gradio can create a public tunnel for you automatically. Share the **Gradio UI**
(port 7860) with:

```bat
python src\run.py --share
```
*(or:* `set SWR_SHARE=1` *then* `python src\run.py` *)*

When it starts you'll see a public URL in the terminal:

```
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

Anyone with that link can open the dashboard. The link lasts while your
terminal is running (it changes each launch).

> ⚠️ Note: `--share` publicises the **Gradio UI on :7860** (the older dashboard).
> To share the newer **RailPulse dashboard on :8000**, use Option B or C below.

---

## Option B — Share the RailPulse dashboard (port 8000) with Cloudflare Tunnel

**No account needed.** Download the one-file `cloudflared.exe` from
https://github.com/cloudflare/cloudflared/releases (latest Windows amd64),
put it in the project folder, then in a **second** cmd window:

```bat
:: first start the app normally
python src\run.py

:: then, in another window:
cloudflared.exe tunnel --url http://localhost:8000
```

It prints a public HTTPS link like `https://random-words.trycloudflare.com`.
Share that link. Free, no signup, works behind home/college Wi-Fi (no port
forwarding).

---

## Option C — Share with ngrok (needs a free account)

1. Sign up at https://ngrok.com and copy your authtoken.
2. `ngrok config add-authtoken <YOUR_TOKEN>`
3. Start the app: `python src\run.py`
4. In another window: `ngrok http 8000`
5. Share the `https://xxxx.ngrok-free.app` link.

---

## Option D — Permanent link: deploy to Hugging Face Spaces (free)

This is the same kind of hosting your reference screenshot used
(`huggingface.co/spaces/...`). It gives a **permanent public URL** that stays up
even when your laptop is off.

1. Create a free account at https://huggingface.co
2. Click **New Space** → name it e.g. `railway-cascade` → SDK: **Gradio** → **Create**.
3. Copy the repo URL, then from this project folder:

```bat
git clone https://huggingface.co/spaces/<YOU>/railway-cascade
:: copy the project into the cloned repo, then:
copy deploy\app.py        railway-cascade\app.py
copy deploy\requirements.txt railway-cascade\requirements.txt
xcopy src  railway-cascade\src\ /E /I
xcopy data railway-cascade\data\ /E /I
xcopy results railway-cascade\results\ /E /I
:: (models/ is optional - see note below)
cd railway-cascade
git add -A && git commit -m "RailPulse cascade dashboard" && git push
```

4. Open `https://huggingface.co/spaces/<YOU>/railway-cascade` — the app boots and
   self-trains if needed (first boot takes ~2-3 min), then the login page appears.

> **Model bundle:** `models/bundle.joblib` is ~32 MB. You may skip committing it —
> the app retrains automatically on the Space. If you want instant boot, commit it
> with Git LFS (`git lfs install && git lfs track "models/*.joblib"`).

> **Login:** default `admin` / `swr2026` (change via Space "Settings → Variables"
> by adding `SWR_USER` and `SWR_PASS`).

---

## Option E — Permanent link: deploy to Render (free tier)

The recommended way to get a permanent public HTTPS URL. Uses the included
**Dockerfile + render.yaml** — no Docker knowledge required.

1. Push the project to a GitHub repo (see `deploy/RENDER.md` for exact commands).
2. Render Dashboard → **New + → Blueprint** → pick the repo → **Apply**.
3. Open `https://<service>.onrender.com` and log in with `admin` / `swr2026`.

**Full step-by-step guide: [`deploy/RENDER.md`](./RENDER.md)** — includes manual
setup, customising credentials, and troubleshooting.
