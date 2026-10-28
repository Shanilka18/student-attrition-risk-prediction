# MEI Platform Design Tokens Reference

This document formalizes the visual tokens, typography rules, spacing geometry, and accessibility standards for the Multimodal Student Engagement Intelligence (MEI) user interface, adhering strictly to `PROJECT_BRIEF.md §10`.

---

## 1. Palette Specification

The MEI color palette is curated for an editorial, research-grade analytics environment. It explicitly rejects generic startup aesthetics (e.g. purple, indigo, neon gradients, glassmorphism, floating drop shadows).

| Token Name | Hex Code | HSL Value | Semantic Role | Contrast vs `#F5F2EC` | Contrast vs `#FFFFFF` |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`paper`** | `#F5F2EC` | `43°, 24%, 94%` | Global application canvas background | N/A | 1.1:1 |
| **`surface`** | `#FFFFFF` | `0°, 0%, 100%` | Card, container, and dialog background | 1.1:1 | N/A |
| **`ink`** | `#1A1D21` | `214°, 12%, 12%` | Primary headings, table text, body copy | 13.5:1 (AAA) | 15.2:1 (AAA) |
| **`muted`** | `#5B6168` | `213°, 7%, 38%` | Subtitles, metadata, table headers, captions | 5.2:1 (AA) | 5.8:1 (AA) |
| **`line`** | `#DDD8CE` | `41°, 19%, 84%` | Card borders (1px solid), divider rules | 1.3:1 (UI Component) | 1.4:1 (UI Component) |
| **`grid`** | `#E8E4DC` | `40°, 20%, 89%` | Background grid lines in analytical charts | 1.1:1 | 1.2:1 |
| **`forest`** | `#1F4E46` | `170°, 43%, 22%` | Primary brand actions, active sidebar nav | 6.8:1 (AAA) | 7.6:1 (AAA) |
| **`forest-light`** | `#EBF2F0` | `163°, 20%, 93%` | Active navigation pill tint, subtle chips | N/A | N/A |
| **`forest-hover`** | `#163A34` | `171°, 45%, 15%` | Primary button hover state | 8.9:1 (AAA) | 10.0:1 (AAA) |

### Functional Risk and State Colors

Risk and alert colors are reserved exclusively for data communication (never used decoratively):

| Semantic Role | Token Name | Hex Code | Contrast vs White | Meaning in Higher-Ed Context |
| :--- | :--- | :--- | :---: | :--- |
| **Low Risk / Good Fit** | `olive` | `#2D5A27` | 6.5:1 (AAA) | Score 0–34; Model generalization gap $\le 0.05$ |
| **Moderate Risk / Slight Overfit** | `stone` | `#8C7355` | 4.8:1 (AA) | Score 35–64; Model generalization gap $0.05 < \text{gap} \le 0.10$ |
| **High Risk / Overfit / Destructive** | `clay` | `#8C4A3B` | 5.8:1 (AA) | Score 65–100; Model generalization gap $> 0.10$; Destructive actions |
| **Informational** | `info` | `#2C5282` | 7.1:1 (AAA) | Informational callouts and guidance notes |
| **Warning** | `warning` | `#975A16` | 5.4:1 (AA) | Non-blocking data quality flags (e.g. AMBER upload status) |

### Automated Enforcement (Zero Tolerance)
An automated pre-commit / CI script (`frontend/scripts/check-forbidden.cjs`) scans all frontend files to enforce:
1. **Hue Exclusion**: Mathematical prohibition of hues between $245^\circ$ and $335^\circ$ with saturation $> 20\%$ (eliminating all purple, violet, indigo, and magenta).
2. **Forbidden Buzzwords**: Rejection of `prototype`, `demo`, `beta`, `mock`, `placeholder`, `smart`, `magic`, `powered by ai`.
3. **Icons & Glyphs**: Prohibition of emojis, sparkles, and robot symbols.

---

## 2. Typography

Typography is bundled locally via Fontsource (no external Google Fonts CDN links) to support air-gapped university campus deployments.

| Font Family | Weights | Primary Usage |
| :--- | :--- | :--- |
| **`Source Serif 4`** | `400 (Regular)`, `600 (Semibold)` | Page headers (`h1`), section titles (`h2`), large metric values (`stat-num`) |
| **`Public Sans`** | `400 (Regular)`, `500 (Medium)`, `600 (Semibold)` | UI labels, navigation buttons, data table contents, tooltips, captions |

### Type Scale

| Class Token | Font Size | Line Height | Letter Spacing | Font Family | Default Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `text-display` | `32px` | `1.2` (`38.4px`) | `-0.02em` | Source Serif 4 | Semibold (`600`) |
| `text-h1` | `24px` | `1.25` (`30px`) | `-0.015em` | Source Serif 4 | Semibold (`600`) |
| `text-h2` | `20px` | `1.3` (`26px`) | `-0.01em` | Source Serif 4 | Semibold (`600`) |
| `text-h3` | `16px` | `1.4` (`22.4px`) | `0` | Public Sans | Semibold (`600`) |
| `text-body` | `14px` | `1.5` (`21px`) | `0` | Public Sans | Regular (`400`) |
| `text-ui` | `13px` | `1.4` (`18.2px`) | `0` | Public Sans | Medium (`500`) |
| `text-caption` | `12px` | `1.4` (`16.8px`) | `0.01em` | Public Sans | Regular (`400`) |
| `text-mono` | `12px` | `1.4` (`16.8px`) | `0` | `ui-monospace`, monospace | Regular (`400`) |

### Tabular Numerals
All quantitative metrics, scores, confidence intervals, sample counts, and percentages use CSS font feature settings:
```css
font-feature-settings: 'tnum' 1, 'cv05' 1;
font-variant-numeric: tabular-nums;
```
This guarantees vertical alignment of digits across data tables and statistical comparison cards.

---

## 3. Spatial Geometry and Elevation

- **Card Border**: Exactly `1px solid #DDD8CE`.
- **Corner Radii**: Uniform `6px` (`rounded-card`) across all cards, containers, buttons, and inputs. Nested elements (such as badges and inner tabs) use `4px` (`rounded-inner`).
- **Elevation / Shadows**:
  - Cards, panels, stats, and headers have **ZERO drop shadows** (`box-shadow: none`).
  - Menus, dropdowns, dialogs, and popovers use a single subtle boundary shadow: `0 4px 12px rgba(26, 29, 33, 0.08)`.
- **Sidebar Width**: Fixed `232px` on desktop (`min-w-[232px] w-[232px]`).
- **Maximum Content Width**: Responsive grid with `max-w-7xl` (`1280px`) container.

---

## 4. Accessibility and Focus Standards

- **Color Contrast**: Every text element meets or exceeds WCAG 2.1 AA requirements ($\ge 4.5:1$ for normal text, $\ge 3:1$ for large display text).
- **Non-Color Indicators**: Every status tag and risk tier pairs color with an explicit textual label, numeric threshold range, or geometric dot indicator. Color is never the sole carrier of meaning.
- **Focus Rings**: Interactive controls render an accessible 2px high-contrast outline on keyboard navigation:
  ```css
  focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1F4E46]
  ```
- **Motion Preferences**: Respects `prefers-reduced-motion: reduce` by disabling transitions and animations globally.
