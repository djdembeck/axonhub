package orchestrator

import (
	"context"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/log"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/internal/server/biz"
)

const (
	defaultCostMaxScore = 80.0

	// defaultMaxCostPer1kTokens is the reference maximum cost per 1K tokens
	// used for inverse normalization. Channels costing this much or more score 0.
	// Set to $2.00 to accommodate both per-token and flat-fee pricing modes.
	defaultMaxCostPer1kTokens = 2.0
)

// ChannelPriceProvider provides channel model pricing information.
type ChannelPriceProvider interface {
	GetChannelPrice(ctx context.Context, channelID int, modelID string) (*ent.ChannelModelPrice, error)
}

// CostAwareStrategy prioritizes channels with lower model costs.
// Lower cost channels receive higher scores via inverse normalization.
type CostAwareStrategy struct {
	metricsProvider ChannelPriceProvider
	maxScore        float64
}

// NewCostAwareStrategy creates a new cost-aware load balancing strategy.
func NewCostAwareStrategy(metricsProvider ChannelPriceProvider) *CostAwareStrategy {
	return &CostAwareStrategy{
		metricsProvider: metricsProvider,
		maxScore:        defaultCostMaxScore,
	}
}

func (s *CostAwareStrategy) Name() string {
	return "CostAware"
}

func (s *CostAwareStrategy) Score(ctx context.Context, channel *biz.Channel) float64 {
	modelID := requestedModelFromContext(ctx)
	if modelID == "" {
		return s.maxScore / 2
	}

	price, err := s.metricsProvider.GetChannelPrice(ctx, channel.ID, modelID)
	if err != nil || price == nil {
		return s.maxScore / 2
	}

	cost := extractRepresentativeCost(price.Price)
	if cost < 0 {
		return s.maxScore / 2
	}

	return s.maxScore * clampNormalizedInverse(cost, defaultMaxCostPer1kTokens)
}

func (s *CostAwareStrategy) ScoreWithDebug(ctx context.Context, channel *biz.Channel) (float64, StrategyScore) {
	modelID := requestedModelFromContext(ctx)

	if modelID == "" {
		neutralScore := s.maxScore / 2

		log.Warn(ctx, "CostAwareStrategy: no model ID in context, using neutral score",
			log.Int("channel_id", channel.ID),
			log.String("channel_name", channel.Name),
		)

		return neutralScore, StrategyScore{
			StrategyName: s.Name(),
			Score:        neutralScore,
			Details: map[string]any{
				"reason": "no_model_id_in_context",
			},
		}
	}

	price, err := s.metricsProvider.GetChannelPrice(ctx, channel.ID, modelID)
	if err != nil {
		neutralScore := s.maxScore / 2

		log.Warn(ctx, "CostAwareStrategy: failed to get price, using neutral score",
			log.Int("channel_id", channel.ID),
			log.String("channel_name", channel.Name),
			log.String("model_id", modelID),
			log.Cause(err),
		)

		return neutralScore, StrategyScore{
			StrategyName: s.Name(),
			Score:        neutralScore,
			Details: map[string]any{
				"error":    err.Error(),
				"model_id": modelID,
			},
		}
	}

	if price == nil {
		neutralScore := s.maxScore / 2

		log.Info(ctx, "CostAwareStrategy: no price data for model, using neutral score",
			log.Int("channel_id", channel.ID),
			log.String("channel_name", channel.Name),
			log.String("model_id", modelID),
		)

		return neutralScore, StrategyScore{
			StrategyName: s.Name(),
			Score:        neutralScore,
			Details: map[string]any{
				"reason":   "no_price_data",
				"model_id": modelID,
			},
		}
	}

	cost := extractRepresentativeCost(price.Price)
	if cost < 0 {
		neutralScore := s.maxScore / 2

		log.Info(ctx, "CostAwareStrategy: could not extract cost from pricing, using neutral score",
			log.Int("channel_id", channel.ID),
			log.String("channel_name", channel.Name),
			log.String("model_id", modelID),
		)

		return neutralScore, StrategyScore{
			StrategyName: s.Name(),
			Score:        neutralScore,
			Details: map[string]any{
				"reason":   "no_representative_cost",
				"model_id": modelID,
			},
		}
	}

	component := clampNormalizedInverse(cost, defaultMaxCostPer1kTokens)
	score := s.maxScore * component

	log.Info(ctx, "CostAwareStrategy: calculated cost-based score",
		log.Int("channel_id", channel.ID),
		log.String("channel_name", channel.Name),
		log.String("model_id", modelID),
		log.Float64("cost", cost),
		log.Float64("component", component),
		log.Float64("score", score),
	)

	return score, StrategyScore{
		StrategyName: s.Name(),
		Score:        score,
		Details: map[string]any{
			"model_id":  modelID,
			"cost":      cost,
			"component": component,
			"max_cost":  defaultMaxCostPer1kTokens,
		},
	}
}

// extractRepresentativeCost extracts a single representative cost value from a ModelPrice.
// Returns -1 if no cost can be determined.
// For flat_fee: uses FlatFee value.
// For usage_per_unit: uses completion_tokens unit price.
// For usage_tiered: uses first tier's completion_tokens price.
func extractRepresentativeCost(mp objects.ModelPrice) float64 {
	for _, item := range mp.Items {
		if item.ItemCode != objects.PriceItemCodeCompletion {
			continue
		}

		return extractCostFromPricing(item.Pricing)
	}

	if len(mp.Items) > 0 {
		return extractCostFromPricing(mp.Items[0].Pricing)
	}

	return -1
}

func extractCostFromPricing(p objects.Pricing) float64 {
	switch p.Mode {
	case objects.PricingModeFlatFee:
		if p.FlatFee != nil {
			cost, _ := p.FlatFee.Float64()
			return cost
		}
	case objects.PricingModeUsagePerUnit:
		if p.UsagePerUnit != nil {
			cost, _ := p.UsagePerUnit.Float64()
			return cost
		}
	case objects.PricingModeTiered:
		if p.UsageTiered != nil && len(p.UsageTiered.Tiers) > 0 {
			cost, _ := p.UsageTiered.Tiers[0].PricePerUnit.Float64()
			return cost
		}
	}

	return -1
}
