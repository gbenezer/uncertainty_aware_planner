# Uncertainty-Aware Demand and Inventory Planning Tool Release Plan Details

## Serving Pattern

- The AI system this tool implements will be served to downstream primary users as published data, specifically as a visualization dashboard used to make long term business planning and shorter term inventory stock decions.

## Stages

- Upon a new imputation or forecasting model version, the anticipated pattern for release is shadow release for visualization troubleshooting and then canary release on a store-by-store basis
- The initial shadow release is viewed by internal developers, domain experts, and UI/UX designers for at least 1 week, but not more than 3, and we expect no cases during this time
- The canary release will be rolled out in an exponential doubling fashion every week following the completion of the shadow release; 1 store first week, 2 added stores the second week, 4 added stores the third week, and so on
- We do not have an estimate of the user base size on a per-store scale, so we do not readily have a case estimate for the canary release

## Promotion Gates

- The promotion gate between shadow and canary is the qualitative assessment of the visualizations provided and agreement on imputation and forecasting quality by a body of domain experts in inventory stocking and long term business planning
- The promotion gate between each canary doubling rollout will be based on the absence of immediate issues in released stores, on both an overall product basis as well as low/high demand item disaggregation
  - No sudden decreases in revenue
  - No sudden increases in manual override requests
  - No sudden increases in stockout hour frequency
- The thresholds for these changes are TBD given we do not have users or historical user data to set them

## Rollback

- Shadow stage
  - Trigger: judgement by domain experts that the model(s) are not producing adequate imputation and forecasting performance
  - Metrics: mean absolute scaled error, mean prediction interval width, prediction interval coverage percent, predicted stockout frequency, qualitative human assessment
  - Owner: model maintainers and data pipeline engineers
  - Steps: roll back the underlying imputed data version, then roll back the forecasting data version, then roll back both models to prior versions and re-train

- Canary stages
  - Trigger: decreases in revenue relative to non-released stores, increases in stockout frequency or manual overrides relative to non-released stores, or explicit user feedback
  - Metrics: revenue, stockout frequency, manual override count
  - Owner: store managers, model maintainers, and data pipeline engineers
  - Steps: inform employees at affected stores that models are being updated and to use unaugmented historical data for the coming week, roll back the underlying imputed data version at the affected stores, roll back the forecasting data version at the affected stores, then investigate issues that arose at affected stores
  - Added release gate post-failure: conclude investigation with understood underlying cause(s), resolve causes, and implement corrective and preventative actions prior to re-starting canary release

## User Base Notification

- User information: at released stores, a notice is sent out to inform employees that the data models are being updated as well as a reminder to cross-check any machine prediction against intuition and domain expertise judgement

- User response: at released stores, users can request manual overrides in the system when the system is wrong, triggering a rollback when a threshold is met