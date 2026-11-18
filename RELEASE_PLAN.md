# Uncertainty-Aware Demand and Inventory Planning Tool: Release Plan

## 1. Serving Pattern

The AI system is served to primary users as **published data** through a **visualization dashboard**. Users rely on it for two decisions:

- Long-term business planning
- Shorter-term inventory stocking

## 2. Release Overview

Every new imputation or forecasting model version follows the same two-stage path:

| Stage | Audience | Duration | Exit gate |
| --- | --- | --- | --- |
| **Shadow** | Internal developers, domain experts, UI/UX designers (no store users) | Minimum 1 week, maximum 3 weeks | Expert review (Section 3.1) |
| **Canary** | Stores, added incrementally | Continues until all stores are released | Per-step metric check (Section 3.2) |

**Canary schedule.** Rollout doubles weekly after shadow completes. The number of *newly added* stores per week is 1, 2, 4, 8, and so on, so the *cumulative* number of released stores is 1, 3, 7, 15, and so on.

**Shadow success criterion.** No critical visualization defects or forecast/imputation quality concerns are raised during the shadow period.

**Known gap.** User base size per store is unknown, so the expected number of affected users (and cases) during canary cannot yet be estimated.

## 3. Promotion Gates

### 3.1 Shadow to Canary

Promotion requires both of the following, from a panel of domain experts in inventory stocking and long-term business planning:

1. A qualitative assessment of the visualizations (clarity, usability, correct representation of uncertainty).
2. Agreement that imputation and forecasting quality is acceptable.

### 3.2 Between Canary Steps

Each doubling step is gated on the **absence of immediate issues** in already-released stores. Each of the following is evaluated overall and disaggregated by low-demand and high-demand items:

- No sudden decrease in revenue
- No sudden increase in manual override requests
- No sudden increase in stockout-hour frequency
- No degradation in uncertainty estimation quality

Metrics at released stores are compared against **non-released stores** to separate model effects from seasonality and local events.

> **Thresholds: TBD.** No users or historical user data exist yet to calibrate them. See Open Items.

## 4. Rollback

### 4.1 Shadow Stage

| | |
| --- | --- |
| **Trigger** | Domain experts judge imputation or forecasting performance inadequate |
| **Metrics** | Mean absolute scaled error (MASE); mean prediction interval width; prediction interval coverage (%); predicted stockout frequency; qualitative human assessment |
| **Owners** | Model maintainers; data pipeline engineers |

**Steps (in order):**

1. Roll back the imputed data version.
2. Roll back the forecasting data version.
3. Roll back both models to their prior versions and re-train.

### 4.2 Canary Stages

| | |
| --- | --- |
| **Trigger** | Any of: lower revenue, higher stockout frequency, or higher manual override count relative to non-released stores; explicit user feedback |
| **Metrics** | Revenue; stockout frequency; manual override count; mean prediction interval width; prediction interval coverage (%) |
| **Owners** | Store managers; model maintainers; data pipeline engineers |

**Steps (in order, at affected stores only):**

1. Notify employees that the models are being updated and that, for the coming week, they should use unaugmented historical data.
2. Roll back the imputed data version.
3. Roll back the forecasting data version.
4. Investigate the issues that arose.

**Re-release gate (after any canary failure).** Canary release restarts only after the investigation has:

- identified and understood the underlying cause(s),
- resolved those causes, and
- put corrective and preventive actions in place.

## 5. User Notification and Response

**Notification.** At released stores, employees receive a notice that includes:

- the data models are being updated, and
- a reminder to cross-check machine predictions against their own intuition and domain expertise.

**User response.** Users can request a manual override whenever the system appears wrong. Override requests are tracked as a canary metric, and a rollback is triggered when the override threshold is met (threshold TBD).
