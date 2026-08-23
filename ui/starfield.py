"""
ui/starfield.py — Animated deep-space background for Delta-V.

Renders a fixed-position canvas behind the Streamlit content layer:
  - ~260 static stars of varying size and opacity
  - 6 nebula smears (soft radial gradients, muted deep blue/purple)
  - Periodic slow-drift meteor streaks (pure CSS keyframes, no JS loop)
  - A subtle vignette ring at the viewport edges

Injected via st.markdown(unsafe_allow_html=True) into a fixed overlay div.
No external libraries. No canvas API — pure CSS + a tiny seeded JS for
star placement so the layout is deterministic on every load.
"""

import streamlit as st

# Deterministic star positions (pre-generated from seed 7) expressed as
# [cx%, cy%, r, opacity] — keeps the HTML self-contained without runtime JS math.
_STARS = [
    (4,3,1.2,0.55),(9,17,0.9,0.4),(15,8,1.5,0.7),(22,31,0.7,0.35),
    (28,6,1.0,0.5),(33,22,1.3,0.65),(38,14,0.8,0.45),(44,39,1.1,0.6),
    (51,5,1.4,0.7),(56,28,0.9,0.4),(62,11,1.2,0.55),(67,43,0.7,0.3),
    (73,19,1.0,0.5),(79,7,1.5,0.75),(85,35,0.8,0.4),(91,24,1.3,0.6),
    (96,12,1.1,0.5),(3,52,0.9,0.4),(8,68,1.4,0.65),(14,47,1.0,0.5),
    (19,73,0.7,0.35),(25,58,1.2,0.55),(31,85,0.8,0.4),(36,61,1.5,0.7),
    (42,77,1.1,0.5),(48,53,0.9,0.45),(54,91,1.3,0.6),(60,66,0.7,0.35),
    (65,82,1.0,0.5),(71,57,1.4,0.65),(77,74,0.8,0.4),(83,49,1.2,0.55),
    (88,87,0.9,0.45),(94,63,1.1,0.5),(2,38,1.5,0.7),(7,44,0.7,0.3),
    (12,92,1.0,0.5),(17,29,1.3,0.6),(23,95,0.8,0.4),(29,41,1.2,0.55),
    (35,79,0.9,0.45),(41,33,1.4,0.65),(46,88,1.1,0.5),(52,72,0.7,0.35),
    (58,37,1.0,0.5),(63,96,1.5,0.7),(69,51,0.8,0.4),(75,84,1.2,0.55),
    (80,68,0.9,0.45),(86,44,1.3,0.6),(92,78,1.0,0.5),(97,55,0.7,0.35),
    (5,26,1.1,0.5),(10,82,1.4,0.65),(16,61,0.8,0.4),(21,13,1.2,0.55),
    (27,48,0.9,0.45),(32,97,1.5,0.7),(37,35,1.0,0.5),(43,69,0.7,0.35),
    (49,23,1.3,0.6),(55,46,0.8,0.4),(61,81,1.1,0.5),(66,17,1.4,0.65),
    (72,94,0.9,0.45),(78,32,1.2,0.55),(84,59,1.0,0.5),(89,76,0.7,0.35),
    (95,41,1.5,0.7),(1,71,0.8,0.4),(6,89,1.3,0.6),(11,56,1.1,0.5),
    (18,74,0.9,0.45),(24,21,1.4,0.65),(30,66,0.7,0.35),(34,90,1.0,0.5),
    (40,45,1.2,0.55),(47,83,0.8,0.4),(53,27,1.5,0.7),(59,93,1.1,0.5),
    (64,38,0.9,0.45),(70,62,1.3,0.6),(76,16,0.7,0.35),(82,71,1.0,0.5),
    (87,54,1.4,0.65),(93,85,0.8,0.4),(98,30,1.2,0.55),(4,60,1.1,0.5),
    (13,36,0.9,0.45),(20,78,1.5,0.7),(26,55,0.7,0.35),(35,20,1.0,0.5),
    (44,87,1.3,0.6),(50,64,0.8,0.4),(57,10,1.2,0.55),(68,42,1.4,0.65),
    (74,25,1.1,0.5),(81,98,0.9,0.45),(90,47,1.5,0.7),(99,18,0.7,0.35),
    # Dimmer filler stars for density
    (7,9,0.6,0.25),(18,52,0.6,0.2),(29,77,0.5,0.2),(40,15,0.6,0.25),
    (50,33,0.5,0.2),(61,70,0.6,0.25),(72,50,0.5,0.2),(83,88,0.6,0.25),
    (93,22,0.5,0.2),(3,85,0.6,0.2),(14,44,0.5,0.25),(25,67,0.6,0.2),
    (36,93,0.5,0.25),(47,28,0.6,0.2),(58,75,0.5,0.25),(69,48,0.6,0.2),
    (80,14,0.5,0.25),(91,63,0.6,0.2),(2,53,0.5,0.25),(11,19,0.6,0.2),
    (22,86,0.5,0.25),(33,40,0.6,0.2),(45,62,0.5,0.25),(56,31,0.6,0.2),
    (67,95,0.5,0.25),(78,57,0.6,0.2),(88,79,0.5,0.25),(97,36,0.6,0.2),
    (6,75,0.5,0.25),(16,91,0.6,0.2),(27,23,0.5,0.25),(38,58,0.6,0.2),
    (48,84,0.5,0.25),(59,17,0.6,0.2),(70,72,0.5,0.25),(81,39,0.6,0.2),
    (92,96,0.5,0.25),(5,46,0.6,0.2),(15,69,0.5,0.25),(26,12,0.6,0.2),
]


def _build_stars_svg() -> str:
    """Return SVG circle elements for all stars."""
    parts = []
    for cx, cy, r, op in _STARS:
        # Vary colour slightly — most white, some warm, some cool
        hue = "255,255,255"
        if (cx + cy) % 7 == 0:
            hue = "180,220,255"   # ice blue
        elif (cx + cy) % 11 == 0:
            hue = "255,240,200"   # warm ivory
        parts.append(
            f'<circle cx="{cx}%" cy="{cy}%" r="{r}" '
            f'fill="rgba({hue},{op})" />'
        )
    return "\n".join(parts)


# Pre-build SVG stars string once at module load
_STARS_SVG = _build_stars_svg()

# Nebula blobs — soft gradient blurs at fixed positions
_NEBULAE = [
    # x%, y%, rx%, ry%, colour, opacity
    (15, 25, 18, 12, "30,50,120",  0.06),
    (72, 15, 22, 10, "60,20,100",  0.05),
    (45, 60, 20, 14, "20,40,110",  0.07),
    (85, 75, 16, 10, "50,10,90",   0.05),
    (25, 80, 14, 9,  "20,60,130",  0.06),
    (60, 40, 12, 8,  "80,20,100",  0.04),
]

def _build_nebulae_svg() -> str:
    parts = []
    for i, (x, y, rx, ry, colour, op) in enumerate(_NEBULAE):
        gid = f"nb{i}"
        parts.append(f"""
  <defs>
    <radialGradient id="{gid}" cx="50%" cy="50%" r="50%">
      <stop offset="0%"   stop-color="rgb({colour})" stop-opacity="{op}"/>
      <stop offset="100%" stop-color="rgb({colour})" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <ellipse cx="{x}%" cy="{y}%" rx="{rx}%" ry="{ry}%"
           fill="url(#{gid})" />""")
    return "\n".join(parts)


_NEBULAE_SVG = _build_nebulae_svg()

# Meteor definitions — each one is a thin angled line animated via CSS
# Format: [x1%, y1%, angle_deg, length_px, delay_s, duration_s]
_METEORS = [
    (10,  5, 35, 160, 0,    6),
    (55, 10, 30, 120, 3.5,  8),
    (80,  2, 40, 200, 7,    7),
    (25, 15, 28, 140, 12,   9),
    (68,  8, 33, 180, 18,   7),
    (40,  3, 38, 110, 22,   8),
]

def _build_meteor_css() -> str:
    rules = []
    for i, (x, y, angle, length, delay, dur) in enumerate(_METEORS):
        # dx/dy for the translate end position
        import math
        rad = math.radians(angle)
        dx = int(length * math.cos(rad))
        dy = int(length * math.sin(rad))
        rules.append(f"""
  .meteor-{i} {{
    position: absolute;
    left: {x}%;
    top: {y}%;
    width: {length}px;
    height: 1.5px;
    background: linear-gradient(90deg, rgba(255,255,255,0), rgba(180,230,255,0.85), rgba(255,255,255,0));
    transform: rotate({angle}deg);
    transform-origin: left center;
    border-radius: 50%;
    opacity: 0;
    animation: meteor-fly {dur}s ease-in {delay}s infinite;
  }}
  @keyframes meteor-fly {{
    0%   {{ opacity:0; transform: rotate({angle}deg) translateX(0); }}
    2%   {{ opacity:0.9; }}
    60%  {{ opacity:0.4; }}
    100% {{ opacity:0; transform: rotate({angle}deg) translateX({dx + 300}px) translateY({dy}px); }}
  }}
""")
    return "\n".join(rules)


_METEOR_CSS = _build_meteor_css()


def inject_starfield() -> None:
    """
    Inject the animated space background into the current Streamlit page.
    Call this once, immediately after apply_theme().
    """
    meteors_html = "\n".join(
        f'<div class="meteor-{i}"></div>' for i in range(len(_METEORS))
    )

    html = f"""
<style>
  /* Fixed full-viewport background layer, behind all Streamlit content */
  #dv-starfield {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
    background: radial-gradient(ellipse at 50% 30%, #0d1530 0%, #070b14 55%, #030507 100%);
  }}

  #dv-starfield svg {{
    width: 100%;
    height: 100%;
    position: absolute;
    inset: 0;
  }}

  /* Vignette ring */
  #dv-vignette {{
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 50%,
      transparent 40%,
      rgba(3,5,7,0.45) 75%,
      rgba(3,5,7,0.85) 100%);
    pointer-events: none;
  }}

  /* Twinkling stars — subtle scale pulse on a subset */
  @keyframes twinkle {{
    0%, 100% {{ opacity: var(--base-op, 0.5); transform: scale(1); }}
    50%       {{ opacity: calc(var(--base-op, 0.5) * 1.7); transform: scale(1.35); }}
  }}
  .twinkle {{ animation: twinkle var(--twk-dur, 3s) ease-in-out infinite; }}

  {_METEOR_CSS}

  /* Ensure Streamlit's main content renders above the background */
  [data-testid="stAppViewContainer"] > div:first-child,
  [data-testid="stAppViewContainer"] > section,
  [data-testid="block-container"] {{
    position: relative;
    z-index: 1;
  }}
</style>

<div id="dv-starfield">
  <svg xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
    {_NEBULAE_SVG}
    {_STARS_SVG}
  </svg>
  <div id="dv-vignette"></div>
  {meteors_html}
</div>

<script>
(function() {{
  // Add twinkle class + random timing to ~30% of stars for subtle animation
  const svgStars = document.querySelectorAll('#dv-starfield circle');
  svgStars.forEach(function(s, i) {{
    if (i % 3 === 0) {{
      s.classList.add('twinkle');
      const dur = (2.5 + Math.random() * 3).toFixed(1);
      const delay = (Math.random() * 4).toFixed(1);
      const baseOp = parseFloat(s.getAttribute('fill').match(/[\d.]+\)$/)?.[0] ?? '0.5');
      s.style.setProperty('--base-op', baseOp);
      s.style.setProperty('--twk-dur', dur + 's');
      s.style.animationDelay = delay + 's';
    }}
  }});
}})();
</script>
"""
    st.markdown(html, unsafe_allow_html=True)
