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

	// Channel with flat fee pricing
	ch1Price := createTestFlatFeePrice("0.5") // $0.50 flat fee
	ch2Price := createTestFlatFeePrice("1.5") // $1.50 flat fee

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

	// Compare flat fee vs usage_per_unit at typical usage
	// Flat fee: $0.10, Usage: $0.02/1K tokens * 1000 tokens = $0.02
	ch1Price := createTestFlatFeePrice("0.10")
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

	// At 1000 tokens, usage-based ($0.02) is cheaper than flat fee ($0.10)
	// So usage-based should score higher
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
