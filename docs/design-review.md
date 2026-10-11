# FloodLab AI design review

## Guidance and scope

No project or ancestor `AGENTS.md` was found. The available global
`/Users/bashyaldolraj/.codex/AGENTS.md` is empty. The user requested the
frontend-design and design-taste-frontend skills. The latter excludes dense
dashboards: its audit, copy, accessibility, and redesign guidance applies here,
while native Streamlit controls remain the product foundation.

Reading this as a scientific workspace for research reviewers, with compact
controls and visualization-first hierarchy. Redesign mode: visual overhaul;
preserve the four views, navigation names, URLs, data contracts, and experiments.
DESIGN_VARIANCE = 4, MOTION_INTENSITY = 2, VISUAL_DENSITY = 7.

## Existing audit

- Brand: FloodLab AI; forest teal accent `#197268`; dark green sidebar, ivory main
  area, Georgia display/metric type, Verdana chart text, inconsistent large spacing.
- Routes: root/explorer, comparison, sensors, physics; no deployed SEO baseline,
  analytics, forms, or consent interface. All route slugs stay intact.
- Working elements: four experiment views, reproducible metrics, frame selection,
  downloads, same-scale comparisons, 3D, optional animation, accessible native
  widgets, and measured/synthetic attribution.
- Visual debt: large literary headlines, numbered eyebrows, repeated sidebar
  descriptions, four paragraphs of interpretation around many charts, large blank
  plotting space, and important plots pushed below the first viewport.
- Retire: slogan-style headings, repeated prose, duplicate sidebar explanations,
  decorative footer strip, and large serif metric typography.

## Visual reference research

Inspected the live sites and captured browser screenshots before editing:

1. [NOAA National Water Prediction Service](https://water.noaa.gov/): visualization
   occupies the main canvas; map tools and search sit beside the data. Borrow the
   priority given to the visualization, not its operational forecast claims.
2. [TensorFlow Playground](https://playground.tensorflow.org/): compact parameter
   strip, direct manipulation, nearby output metrics, and explanations below the
   working area. Borrow control/plot adjacency and concise labels.
3. [Copernicus GloFAS](https://global-flood.emergency.copernicus.eu/): clear separation
   between a public information site and the map viewer. Borrow the separation of
   research context from exploration; do not add a marketing carousel to this app.

## Design decisions

- Self-host IBM Plex Sans and IBM Plex Mono with their OFL licenses. Compact sans
  headings, tabular monospaced metrics, and clear native form labels.
- Neutral cool surfaces, teal active state, one 6 px radius scale, thin borders,
  consistent light/dark tokens, and a quieter navigation rail.
- Direct view titles and one short description. Essential units and experimental
  context stay visible; detailed methods, interpretations, and source attribution
  move into disclosures.
- Plots retain actual observations, missing cells, units, and shared scales.
  Graphics are scientific data, so no generated photography or illustrative maps.
- Native controls retain keyboard access and labels. Hover/focus feedback only;
  wave animation stays opt-in.

## Verification

Verified all four desktop views and phone layouts at 390 px. Checked light and
dark appearance, self-hosted font loading, chart labels, disclosures, native
navigation, and no horizontal page overflow. Reference lines remain visible in
dark mode. Primary text and accent/button color pairs exceed WCAG AA 4.5:1.

All eight dashboard checks passed in the lightweight app environment, including
model/wave/frame selection, 3D, animation, sensor budgets, residual selection,
published scientific metrics, and downloaded values. `pip check` and
`git diff --check` passed. The packaged project includes native theme settings,
font files and licenses, all four views, and verified experiment artifacts.

## Responsive layout and typography follow-up

- Cap the centered workspace at 1,568 CSS pixels, with bounded gutters. The
  usable main-column width drives container queries, so the sidebar cannot
  squeeze chart pairs or controls on tablets and small laptops.
- Stack paired plots below 900 px of main-column space and three-map comparisons
  below 1,140 px. Controls use two columns below 900 px and one below 540 px.
  Narrow metric groups keep two columns; the third synthetic metric spans both.
- Use a consistent 16 px body, 14 px labels/captions/notes, 13 px chart ticks and
  legends, 14 px axis titles, 15 px chart titles, and 24–28 px metric values.
  Long disclosure paragraphs have a 75-character reading-width limit.
- Remove doubled sidebar padding so complete navigation labels remain visible.
  Self-hosted Plex Sans and the used Plex Mono weight load with font-display swap.

Verified 28 route/theme/viewport combinations in Chromium, including all four
views in light and dark appearance. Widths span 320–3,840 CSS pixels, with tablet,
laptop, 1080p, and wide-monitor sizes; screenshots use a 2× device pixel ratio.
No horizontal page overflow, clipped metric values or navigation labels, missing
used fonts, Streamlit exceptions, or browser JavaScript errors were detected.
Chart axis and label sizes were checked from the rendered plots. Inspected phone
and wide-screen screenshots, including wrapped legends and dark reference lines.
The eight dashboard interaction/scientific checks still pass.

## Fewer boxed sections

Removed enclosing cards from the control strips and flood maps, and replaced
dataset badges with plain context labels. Controls use one subtle bottom divider;
the About section uses one top divider. Methods, benchmark scores, and animation
remain native expandable rows with no rounded enclosing borders. Export actions
use native tertiary buttons. All sections align to the same main-column edges.
Checked the four views, phone and desktop layouts, and light/dark appearance.
The eight existing dashboard checks pass after these changes.

## Header theme control

Added compact System / Light / Dark options to the native header. Dark is the
default for new visitors; System follows system appearance. Explicit choices use
Streamlit's browser persistence. The
control reports actual appearance to the server so Plotly labels and reference
lines update with native widget colors. Keyboard arrows, focus restoration,
repeated switches, navigation, same-route refresh, and system changes were checked.
Verified the controls fit at 320 px without horizontal overflow and inspected
phone and desktop screenshots. All eight dashboard checks pass.

## Icon toggle and Apple typography

Replaced the three theme labels with one sun/moon button. The button joins the
native header row beside the main menu instead of using a fixed right offset;
GitHub and Share actions keep their own layout space. Its accessible label and
tooltip name the next theme. Enter and Space retain native button behavior.

Use the Apple system font stack for the interface and Plotly figures, with
tabular metric numerals. San Francisco is provided by Apple devices; no font
files are redistributed or fetched. Removed the unused Plex font assets.

Checked light/dark switching and focus in the local browser. At 320 px, injected
GitHub/Share-sized toolbar controls remained separate from the toggle and main
menu, with no horizontal page overflow. This was a local layout check; the live
Cloud header was not accessible at the supplied short app name. Eight dashboard
checks pass.

## Visitor context and reconstruction error

Added the visitor introduction below Flood Explorer, three metric help tooltips,
and a third map showing absolute prediction error in millimeters. Measured and
predicted maps share their elevation scale; error uses a separate scale fixed
across the selected model's frames. Maps stack on narrow screens. Sidebar links
have a clear Project links label, with a short About This Research disclosure.

Verified the error map against displayed measurements and predictions at 1,151
locations: its mean is 2.47591657 mm, matching the displayed 2.48 mm MAE. Checked
the RMSE tooltip, desktop three-map layout, phone stacking, and no horizontal
overflow. The eight existing dashboard checks pass across all models and waves.
