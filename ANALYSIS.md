# Captcha Visual Analysis

## Image Properties
- Size: 400x76 px PNG
- Background: solid pastel color (varies — light green, light blue, pink, beige)
- Characters: 4-8 per image, alphanumeric (a-z, 0-9), case-insensitive

## Character Rendering
- Multiple fancy/decorative fonts, some with 3D/emboss/shadow effects
- Each character independently: colored, rotated (up to ~30deg), vertically offset
- Colors vary per character — dark blues, greens, browns, reds, purples
- Some characters have outlines or drop shadows in contrasting colors
- Character spacing is loose and irregular

## Noise Overlay
- Thin colored lines (1-2px) crossing the image at random angles
- Colored ellipses/circles (outlined, not filled) of varying sizes
- Some filled semi-transparent circles/ellipses in pastel colors
- Noise colors span the full spectrum — pinks, greens, blues, yellows
- Noise overlaps characters but characters remain visually distinguishable to humans

## Key Observations
- The noise is geometrically simple (lines + ellipses) but dense
- Characters are thicker/bolder than noise lines
- The 3D font effects make characters look "chunky" — this is actually helpful for distinguishing from thin noise
- Background is always light/pastel, characters are always darker/more saturated
- Character set appears to be lowercase a-z + digits 0-9 (no uppercase in ground truth, though rendering uses mixed case visually)
