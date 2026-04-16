package orchestrator

import (
	"context"
	"testing"

	"github.com/shopspring/decimal"
	"github.com/stretchr/testify/assert"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/internal/server/biz"
)

func TestCostAwareStrategy_Name(t *testing.T) {
	strategy := NewCostAwareStrategy(&mockChannelPriceProvider{
		prices: make(map[int]map[string]*ent.ChannelModelPrice),
	})
	assert.Equal(t, "CostAware", strategy.Name())
}

func TestCostAwareStrategy_Score_NoPriceData_ReturnsNeutral(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {}, // empty price map for channel 1
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	channel := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "ch1"}}
	score := strategy.Score(ctx, channel)
	assert.Equal(t, defaultCostMaxScore/2, score)
}

func TestCostAwareStrategy_Score_CheaperChannelScoresHigher(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	// Channel 1: $0.01 per 1K completion tokens
	// Channel 2: $0.03 per 1K completion tokens
	ch1Price := createTestPrice(objects.PricingModeUsagePerUnit, "0.01")
	ch2Price := createTestPrice(objects.PricingModeUsagePerUnit, "0.03")

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"gpt-4": ch1Price},
			2: {"gpt-4": ch2Price},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	ch1 := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "cheap"}}
	ch2 := &biz.Channel{Channel: &ent.Channel{ID: 2, Name: "expensive"}}

	scoreCh1 := strategy.Score(ctx, ch1)
	scoreCh2 := strategy.Score(ctx, ch2)

	assert.Greater(t, scoreCh1, scoreCh2, "cheaper channel should score higher")
}

func TestCostAwareStrategy_Score_MoreExpensiveChannelScoresLower(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	ch1Price := createTestPrice(objects.PricingModeUsagePerUnit, "0.05")
	ch2Price := createTestPrice(objects.PricingModeUsagePerUnit, "0.01")

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"gpt-4": ch1Price},
			2: {"gpt-4": ch2Price},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	ch1 := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "expensive"}}
	ch2 := &biz.Channel{Channel: &ent.Channel{ID: 2, Name: "cheap"}}

	scoreCh1 := strategy.Score(ctx, ch1)
	scoreCh2 := strategy.Score(ctx, ch2)

	assert.Less(t, scoreCh1, scoreCh2, "more expensive channel should score lower")
}

func TestCostAwareStrategy_Score_FlatFeePricing(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	// Flat fee pricing within the realistic range (under $0.10/1K)
	ch1Price := createTestFlatFeePrice("0.02")
	ch2Price := createTestFlatFeePrice("0.08")

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"gpt-4": ch1Price},
			2: {"gpt-4": ch2Price},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	ch1 := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "cheap-flat"}}
	ch2 := &biz.Channel{Channel: &ent.Channel{ID: 2, Name: "expensive-flat"}}

	scoreCh1 := strategy.Score(ctx, ch1)
	scoreCh2 := strategy.Score(ctx, ch2)

	assert.Greater(t, scoreCh1, scoreCh2, "lower flat fee should score higher")
}

func TestCostAwareStrategy_Score_UsagePerUnitPricing(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	// Test usage_per_unit pricing mode with completion tokens
	ch1Price := createTestPrice(objects.PricingModeUsagePerUnit, "0.002") // $2/1K tokens
	ch2Price := createTestPrice(objects.PricingModeUsagePerUnit, "0.010") // $10/1K tokens

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"gpt-4": ch1Price},
			2: {"gpt-4": ch2Price},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	ch1 := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "cheap-per-unit"}}
	ch2 := &biz.Channel{Channel: &ent.Channel{ID: 2, Name: "expensive-per-unit"}}

	scoreCh1 := strategy.Score(ctx, ch1)
	scoreCh2 := strategy.Score(ctx, ch2)

	assert.Greater(t, scoreCh1, scoreCh2, "lower per-unit price should score higher")
}

func TestCostAwareStrategy_Score_UsageTieredPricing(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	// Test tiered pricing - first channel has cheaper first tier
	ch1Price := createTestTieredPrice([]string{"0.01", "0.02"}, []int{1000, 2000})
	ch2Price := createTestTieredPrice([]string{"0.03", "0.04"}, []int{1000, 2000})

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"gpt-4": ch1Price},
			2: {"gpt-4": ch2Price},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	ch1 := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "cheap-tiered"}}
	ch2 := &biz.Channel{Channel: &ent.Channel{ID: 2, Name: "expensive-tiered"}}

	scoreCh1 := strategy.Score(ctx, ch1)
	scoreCh2 := strategy.Score(ctx, ch2)

	assert.Greater(t, scoreCh1, scoreCh2, "cheaper tiered pricing should score higher")
}

func TestCostAwareStrategy_Score_MixedPricingModes(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	// Compare flat fee vs usage_per_unit: raw dollar values are used directly
	// without token-count normalization (see extractRepresentativeCost docs).
	ch1Price := createTestFlatFeePrice("0.08")
	ch2Price := createTestPrice(objects.PricingModeUsagePerUnit, "0.002")

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"gpt-4": ch1Price},
			2: {"gpt-4": ch2Price},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	ch1 := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "flat-fee"}}
	ch2 := &biz.Channel{Channel: &ent.Channel{ID: 2, Name: "usage-based"}}

	scoreCh1 := strategy.Score(ctx, ch1)
	scoreCh2 := strategy.Score(ctx, ch2)

	// Flat fee of $0.10 scores 0 (exactly at the max reference), per-unit $0.002/1K scores high.
	// Per-unit should score higher because its representative cost is lower.
	assert.Greater(t, scoreCh2, scoreCh1, "usage-based should score higher at low token count")
}

func TestCostAwareStrategy_Score_EmptyModelID_ReturnsNeutral(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "") // empty model ID

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"gpt-4": createTestPrice(objects.PricingModeUsagePerUnit, "0.01")},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	channel := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "ch1"}}
	score := strategy.Score(ctx, channel)
	assert.Equal(t, defaultCostMaxScore/2, score)
}

func TestCostAwareStrategy_Score_ZeroCost_ReturnsMaxScore(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "free-model")

	freePrice := createTestPrice(objects.PricingModeUsagePerUnit, "0")

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"free-model": freePrice},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	channel := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "free-channel"}}
	score := strategy.Score(ctx, channel)
	assert.Equal(t, defaultCostMaxScore, score, "zero cost should yield maximum score")
}

func TestCostAwareStrategy_Score_CostExceedingMax_ReturnsZero(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "expensive-model")

	// Cost of $0.50/1K exceeds defaultMaxCostPer1kTokens ($0.10), so score should be 0.
	expensivePrice := createTestPrice(objects.PricingModeUsagePerUnit, "0.50")

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"expensive-model": expensivePrice},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	channel := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "ultra-expensive"}}
	score := strategy.Score(ctx, channel)
	assert.Equal(t, 0.0, score, "cost exceeding max should yield zero score")
}

func TestCostAwareStrategy_Score_ProviderError_ReturnsNeutral(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "gpt-4")

	mockProvider := &mockChannelPriceProvider{
		err: assert.AnError,
	}
	strategy := NewCostAwareStrategy(mockProvider)

	channel := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "ch1"}}
	score := strategy.Score(ctx, channel)
	assert.Equal(t, defaultCostMaxScore/2, score, "provider error should yield neutral score")
}

func TestCostAwareStrategy_Score_PromptOnlyPricing_ReturnsNeutral(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "prompt-only-model")

	// Model with only prompt_tokens pricing, no completion_tokens item.
	promptOnlyPrice := &ent.ChannelModelPrice{
		ModelID: "prompt-only-model",
		Price: objects.ModelPrice{
			Items: []objects.ModelPriceItem{
				{
					ItemCode: objects.PriceItemCodeUsage,
					Pricing: objects.Pricing{
						Mode:         objects.PricingModeUsagePerUnit,
						UsagePerUnit: func() *decimal.Decimal { d, _ := decimal.NewFromString("0.01"); return &d }(),
					},
				},
			},
		},
	}

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"prompt-only-model": promptOnlyPrice},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	channel := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "ch1"}}
	score := strategy.Score(ctx, channel)

	// Falls back to Items[0] as representative cost since no completion item exists.
	// Score should be > neutral since $0.01 is well under $0.10 max.
	assert.Greater(t, score, defaultCostMaxScore/2, "prompt-only pricing should fall back to first item, not yield neutral")
}

func TestCostAwareStrategy_Score_NoItems_ReturnsNeutral(t *testing.T) {
	ctx := contextWithRequestedModel(context.Background(), "empty-model")

	// Model with no pricing items at all.
	emptyPrice := &ent.ChannelModelPrice{
		ModelID: "empty-model",
		Price:   objects.ModelPrice{Items: nil},
	}

	mockProvider := &mockChannelPriceProvider{
		prices: map[int]map[string]*ent.ChannelModelPrice{
			1: {"empty-model": emptyPrice},
		},
	}
	strategy := NewCostAwareStrategy(mockProvider)

	channel := &biz.Channel{Channel: &ent.Channel{ID: 1, Name: "ch1"}}
	score := strategy.Score(ctx, channel)
	assert.Equal(t, defaultCostMaxScore/2, score, "no pricing items should yield neutral score")
}

// Helper functions for creating test prices

func createTestPrice(mode objects.PricingMode, pricePerUnit string) *ent.ChannelModelPrice {
	decPrice, _ := decimal.NewFromString(pricePerUnit)
	return &ent.ChannelModelPrice{
		ModelID: "gpt-4",
		Price: objects.ModelPrice{
			Items: []objects.ModelPriceItem{
				{
					ItemCode: objects.PriceItemCodeCompletion,
					Pricing: objects.Pricing{
						Mode:         mode,
						UsagePerUnit: &decPrice,
					},
				},
			},
		},
	}
}

func createTestFlatFeePrice(flatFee string) *ent.ChannelModelPrice {
	decFee, _ := decimal.NewFromString(flatFee)
	return &ent.ChannelModelPrice{
		ModelID: "gpt-4",
		Price: objects.ModelPrice{
			Items: []objects.ModelPriceItem{
				{
					ItemCode: objects.PriceItemCodeCompletion,
					Pricing: objects.Pricing{
						Mode:    objects.PricingModeFlatFee,
						FlatFee: &decFee,
					},
				},
			},
		},
	}
}

func createTestTieredPrice(prices []string, thresholds []int) *ent.ChannelModelPrice {
	tiers := make([]objects.PriceTier, len(prices))
	for i, p := range prices {
		decPrice, _ := decimal.NewFromString(p)
		threshold := int64(thresholds[i])
		tiers[i] = objects.PriceTier{
			UpTo:         &threshold,
			PricePerUnit: decPrice,
		}
	}
	tieredPricing := &objects.TieredPricing{
		Tiers: tiers,
	}
	return &ent.ChannelModelPrice{
		ModelID: "gpt-4",
		Price: objects.ModelPrice{
			Items: []objects.ModelPriceItem{
				{
					ItemCode: objects.PriceItemCodeCompletion,
					Pricing: objects.Pricing{
						Mode:        objects.PricingModeTiered,
						UsageTiered: tieredPricing,
					},
				},
			},
		},
	}
}

// mockChannelPriceProvider implements the price provider interface for testing
type mockChannelPriceProvider struct {
	prices map[int]map[string]*ent.ChannelModelPrice
	err    error
}

func (m *mockChannelPriceProvider) GetChannelPrice(ctx context.Context, channelID int, modelID string) (*ent.ChannelModelPrice, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.prices == nil {
		return nil, nil
	}
	channelPrices, ok := m.prices[channelID]
	if !ok {
		return nil, nil
	}
	price, ok := channelPrices[modelID]
	if !ok {
		return nil, nil
	}
	return price, nil
}
