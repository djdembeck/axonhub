package orchestrator

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm/httpclient"
)

var ErrLocalRPMExhausted = errors.New("local channel rpm exhausted")

// LocalRPMExhaustedError represents a local per-channel RPM admission failure.
//
// The embedded synthetic 429 lets existing response transformation produce a
// rate-limit-shaped client error, while the typed wrapper lets retry and
// cooldown code distinguish it from provider 429 responses.
type LocalRPMExhaustedError struct {
	ChannelID   int
	ChannelName string
	Limit       int64

	httpErr *httpclient.Error
}

func newLocalRPMExhaustedError(ch *biz.Channel, limit int64) *LocalRPMExhaustedError {
	message := fmt.Sprintf("channel %s exhausted local RPM limit %d; please retry shortly", ch.Name, limit)
	body := fmt.Appendf(nil, `{"error":{"code":"channel_rpm_exhausted","message":%q,"type":"rate_limit_error"}}`, message)

	return &LocalRPMExhaustedError{
		ChannelID:   ch.ID,
		ChannelName: ch.Name,
		Limit:       limit,
		httpErr: &httpclient.Error{
			StatusCode: http.StatusTooManyRequests,
			Status:     http.StatusText(http.StatusTooManyRequests),
			Body:       body,
		},
	}
}

func (e *LocalRPMExhaustedError) Error() string {
	return fmt.Sprintf("channel %q (id=%d) local rpm exhausted", e.ChannelName, e.ChannelID)
}

func (e *LocalRPMExhaustedError) Unwrap() []error {
	return []error{ErrLocalRPMExhausted, e.httpErr}
}

func isLocalRPMExhaustedError(err error) bool {
	var rpmErr *LocalRPMExhaustedError
	return errors.As(err, &rpmErr)
}

// isRouterCapacityError reports a 429 produced by the local LLM router's
// admission/capacity gates, identified by its X-Router-Reason response header.
// The router is healthy but its local queue is full; the request should fall
// back, but the channel must not be cooled down like a provider quota 429.
func isRouterCapacityError(err error) bool {
	var httpErr *httpclient.Error
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusTooManyRequests {
		return false
	}
	return httpErr.Headers != nil && httpErr.Headers.Get("X-Router-Reason") != ""
}
