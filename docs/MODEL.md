# Model

DriveWorld is a BEV occupancy diffusion world model. It predicts future occupancy and occupancy flow before planning on the predicted future.

## Inputs

```text
past_bev              historical BEV occupancy
map_bev               rasterized lane / road-line / road-edge priors
map_vectors           vectorized lane graph tokens
traffic_lights        historical traffic-light states
agent_features        historical dynamic-agent states
sensor_context        optional camera/LiDAR token context
```

## Outputs

```text
future_occ            future BEV occupancy probabilities
future_flow           future occupancy motion field
multi_sample_future   diffusion samples for uncertainty
ego_plan_risk         risk scores from planning-on-prediction
```

## Architecture

```text
noisy future BEV target
        |
        v
conditional BEV denoiser
  - raster condition encoder
  - temporal BEV encoder
  - vector map encoder
  - traffic-light encoder
  - agent token encoder
  - optional sensor token encoder
  - cross-attention BEV fusion
        |
        v
denoised future occupancy / flow
```

The diffusion scheduler is DDIM. The model can be trained with `prediction_type: sample` or `prediction_type: epsilon`, with `sample` used by the released configuration.

## Planning-On-Prediction

The diffusion model does not take ego plans as input. Ego planning is a downstream consumer:

```text
predicted future occupancy + uncertainty + map prior
        |
        v
lane-following candidate generation
        |
        v
risk scoring
  collision cost
  uncertainty cost
  off-road cost
  smoothness cost
  progress reward
        |
        v
lowest-risk ego plan
```

This keeps the project focused on a world model rather than a plan-conditioned simulator.
