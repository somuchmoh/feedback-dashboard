# Feedback Theme Taxonomy

## Purpose

This taxonomy is the human-reviewed reference used to evaluate whether automated clustering groups feedback into meaningful product themes. The existing `product_area` column is useful context, but it is not ground truth: it may be noisy, overly broad, or based on the team that submitted the feedback.

## Labelling rules

1. Choose the most specific established theme that captures the main product problem, request, or user need.
2. Add a `secondary_theme` only when the feedback contains a second independently meaningful subject or use case. A blank secondary theme is valid and preferred over a weak assignment.
3. Do not copy a submitting team into a theme. Values such as `internal_marketing`, `internal_design`, `internal_sales`, `internal_support`, `testing`, and `customer` belong in `source`.
4. Do not automatically copy `product_area`. Use the feedback text as the primary evidence and treat product area as supporting context.
5. Avoid pairing a specific theme with its broad parent solely to fill the secondary field. For example, `editor_reliability` does not automatically require `editor`.
6. If no theme fits, set `is_outlier` to `true`, use `low` confidence, and explain the gap in `review_notes`. Add a new category only when it is likely to recur.
7. Use lowercase snake_case values exactly as written below.

## Confidence scale

- `high`: The text directly and unambiguously supports the label.
- `medium`: The label is the best available fit, but another interpretation is plausible or the taxonomy is still broad.
- `low`: The fit is weak, the text lacks context, or the row may be an outlier.

## Controlled themes

### analytics

Usage measurement, reporting, trends, metrics, and insights used to understand adoption, behaviour, or outcomes.

Include: page views, edit activity, usage trends, actionable reporting.

Exclude: visual dashboard layout problems without a measurement need; use `dashboard`.

### automation_workflows

Rules, triggers, reminders, templates, integrations, or automated actions that reduce manual work.

Include: location-triggered reminders, combining filters with templates or automation.

Exclude: ordinary filtering or relation navigation with no automation component; use `editor` or `search` as appropriate.

### collaboration

Multi-user editing, coordination, activity awareness, change attribution, and collaboration notifications.

Include: concurrent editing delays, active-editor presence, noisy collaboration activity.

Exclude: access-control decisions; use `permissions`.

### dashboard

Dashboard structure, visual hierarchy, layout, customization, presentation, and at-a-glance comprehension.

Include: cluttered dashboards, executive readability, dashboard screenshots, layout scaling.

Exclude: missing measurements or reporting logic; use `analytics`.

### editor

Content creation and manipulation, databases, relations, formatting controls, commands, and editor navigation.

Include: toolbar density, database relations, custom shortcuts, text-formatting capabilities.

Exclude: crashes, save failures, or severe degradation; use `editor_reliability`.

### editor_reliability

Failures or degraded behaviour while creating or editing content.

Include: save failures, crashes, broken paste behaviour, disappearing editor elements, severe slowdowns on complex pages.

Exclude: confusing controls or missing capabilities that otherwise work; use `editor` or `support_ux`.

### marketing_initiative

Product-surface requests whose purpose is feature promotion, launch communication, or adoption campaigns.

Include: promotional banners, announcement components, in-product feature callouts.

Exclude: feedback merely submitted by marketing; use the actual product theme and retain marketing in `source`.

### mobile_ui

Mobile-specific navigation, layout, touch interaction, feature parity, orientation, or input capabilities.

Include: cramped mobile tables, missing mobile controls, stylus support, mobile navigation.

Exclude: a feature failure whose main subject is version history or another specific system; use that feature as primary and `mobile_ui` as secondary.

### onboarding

First-use learning, conceptual understanding, setup guidance, defaults, and role-specific introduction.

Include: pages-versus-databases confusion, generic tutorials, unclear workspace-structure guidance.

Exclude: ongoing usability problems experienced after onboarding; use the affected product theme.

### permissions

Access control, sharing roles, password protection, permission inheritance, auditing, and accidental access changes.

Include: view-versus-edit confusion, granular client access, workspace permission audits.

Exclude: change history and recovery; use `versioning`.

### sales_enablement

Capabilities or presentation needs explicitly intended to support demos, renewals, buyer reassurance, or sales conversations. Prefer this as a secondary theme when a more specific product capability is clear.

Include: renewal evidence, simplified demo workspaces, buyer-facing feature visibility.

Exclude: feedback merely submitted by sales; use the actual product theme and retain sales in `source`.

### search

Information retrieval, relevance, indexing, filtering search results, discoverability of older content, and search performance.

Include: irrelevant results, indexing delays, difficulty finding older documents.

Exclude: navigating between already-known related pages; use `workspace` or `editor`.

### support_ux

Cross-cutting clarity or guidance problems that cause avoidable confusion and support demand. Prefer this as a secondary theme when a specific product area is identifiable.

Include: unclear labels, missing inline tips, confusing error explanations.

Exclude: general dissatisfaction without a concrete clarity or guidance issue.

### versioning

Version history, revision comparison, recovery, named versions, and visibility of past changes.

Include: hard-to-scan revision history, unclear recovery, missing comparisons.

Exclude: general collaboration attribution without a version-recovery need; use `collaboration`.

### workspace

Workspace-level navigation, synchronization, scale, configuration, integrations, and cross-page organization.

Include: large-workspace performance, sync delays, workspace-wide organization.

Exclude: search-specific scaling problems; use `search` and optionally `workspace` as secondary.

## Labels that are not themes

The following values describe provenance or organizational ownership and must not be used as primary or secondary themes:

- `customer`
- `testing`
- `internal_design`
- `internal_marketing`
- `internal_sales`
- `internal_support`

## Secondary-theme examples

- Version history fails only on mobile: `versioning` + `mobile_ui`.
- Usage analytics needed for renewals: `analytics` + `sales_enablement`.
- Dashboard clutter hurts executive demos: `dashboard` + `sales_enablement`.
- Permission labels confuse users: `permissions` + `support_ux`.
- A clear single-topic request: primary theme plus a blank secondary theme.
