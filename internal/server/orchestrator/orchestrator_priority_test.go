package orchestrator

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/looplj/axonhub/internal/server/biz"
)

func TestSelectLoadBalancer_AdaptiveCost(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	lb := processor.selectLoadBalancer(context.Background(), biz.LoadBalancerStrategyAdaptive, biz.LoadBalancerPriorityCost)

	assert.Same(t, processor.costPriorityLB, lb, "adaptive+cost should select costPriorityLB")
}

func TestSelectLoadBalancer_AdaptiveTPS(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	lb := processor.selectLoadBalancer(context.Background(), biz.LoadBalancerStrategyAdaptive, biz.LoadBalancerPriorityTPS)

	assert.Same(t, processor.tpsPriorityLB, lb, "adaptive+tps should select tpsPriorityLB")
}

func TestSelectLoadBalancer_AdaptiveTTFT(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	lb := processor.selectLoadBalancer(context.Background(), biz.LoadBalancerStrategyAdaptive, biz.LoadBalancerPriorityTTFT)

	assert.Same(t, processor.ttftPriorityLB, lb, "adaptive+ttft should select ttftPriorityLB")
}

func TestSelectLoadBalancer_AdaptiveEmptyPriorityDefaultsToTTFT(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	lb := processor.selectLoadBalancer(context.Background(), biz.LoadBalancerStrategyAdaptive, "")

	assert.Same(t, processor.ttftPriorityLB, lb, "adaptive+empty priority should default to ttftPriorityLB")
}

func TestSelectLoadBalancer_AdaptiveUnknownPriorityDefaultsToTTFT(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	lb := processor.selectLoadBalancer(context.Background(), biz.LoadBalancerStrategyAdaptive, "unknown_priority")

	assert.Same(t, processor.ttftPriorityLB, lb, "adaptive+unknown priority should default to ttftPriorityLB")
}

func TestSelectLoadBalancer_FailoverIgnoresPriority(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	for _, priority := range []string{biz.LoadBalancerPriorityCost, biz.LoadBalancerPriorityTPS, biz.LoadBalancerPriorityTTFT, ""} {
		lb := processor.selectLoadBalancer(context.Background(), biz.LoadBalancerStrategyFailover, priority)

		assert.Same(t, processor.failoverLoadBalancer, lb,
			"failover strategy should always select failoverLoadBalancer regardless of priority (priority=%q)", priority)
	}
}

func TestSelectLoadBalancer_CircuitBreakerIgnoresPriority(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	for _, priority := range []string{biz.LoadBalancerPriorityCost, biz.LoadBalancerPriorityTPS, biz.LoadBalancerPriorityTTFT, ""} {
		lb := processor.selectLoadBalancer(context.Background(), biz.LoadBalancerStrategyCircuitBreaker, priority)

		assert.Same(t, processor.circuitBreakerLoadBalancer, lb,
			"circuit-breaker strategy should always select circuitBreakerLoadBalancer regardless of priority (priority=%q)", priority)
	}
}

func TestSelectLoadBalancer_UnknownStrategyDefaultsToTTFT(t *testing.T) {
	processor := newProcessorWithDistinctLBs()

	lb := processor.selectLoadBalancer(context.Background(), "unknown_strategy", biz.LoadBalancerPriorityCost)

	assert.Same(t, processor.ttftPriorityLB, lb, "unknown strategy should default to ttftPriorityLB")
}

// newProcessorWithDistinctLBs creates a ChatCompletionOrchestrator with distinct
// LoadBalancer pointers so tests can verify selection via pointer identity.
// Note: nil systemService/channelService is safe here because tests only verify
// pointer identity via assert.Same — the LoadBalancers are never exercised.
func newProcessorWithDistinctLBs() *ChatCompletionOrchestrator {
	return &ChatCompletionOrchestrator{
		ttftPriorityLB:             NewLoadBalancer(nil, nil),
		tpsPriorityLB:              NewLoadBalancer(nil, nil),
		costPriorityLB:             NewLoadBalancer(nil, nil),
		failoverLoadBalancer:       NewLoadBalancer(nil, nil),
		circuitBreakerLoadBalancer: NewLoadBalancer(nil, nil),
	}
}
