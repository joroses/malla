/**
 * RF link quality UI helpers (see LINK_QUALITY_CONTRACT.md).
 *
 * Shared DOM formatting for per-direction RF link metrics. Every metric,
 * quality tier, color and balance classification is rendered from the
 * backend link payload (`forward_*` / `return_*` / `link_balance` /
 * `is_bidirectional` fields produced by `enrich_link_quality`); this helper
 * never recomputes quality thresholds, colors or reliability curves, so the
 * map, graph and link pages cannot drift apart.
 *
 * The palette itself (including the unknown fallback) comes from the
 * backend-injected `window.MALLA_QUALITY_COLORS` (see base.html); no hex
 * value is duplicated here.
 *
 * Requires dom.js (`el`, `textNode`) to be loaded first.
 */
(function () {
    function qualityColor(quality) {
        const palette = window.MALLA_QUALITY_COLORS || {};
        return palette[quality || 'unknown'] || palette.unknown;
    }

    function hasValue(value) {
        return value !== null && value !== undefined && !Number.isNaN(value);
    }

    function formatDb(value) {
        return hasValue(value) ? `${value.toFixed(1)} dB` : null;
    }

    function formatDbm(value) {
        return hasValue(value) ? `${value.toFixed(1)} dBm` : null;
    }

    function formatPercent(value) {
        return hasValue(value) ? `${value.toFixed(1)}%` : null;
    }

    /**
     * Badge for a quality tier. The tier text and its color both come from
     * the backend payload; only the capitalization is cosmetic. A missing
     * payload color falls back to the backend palette's unknown color.
     */
    function qualityBadge(quality, color) {
        const label = quality
            ? quality.charAt(0).toUpperCase() + quality.slice(1)
            : 'Unknown';
        return el('span', {
            className: 'badge rf-quality-badge',
            style: { backgroundColor: color || qualityColor() }
        }, label);
    }

    /**
     * One directional measurement block for a link popup.
     *
     * Reads `${prefix}_avg_snr`, `${prefix}_avg_rssi`,
     * `${prefix}_count`, `${prefix}_quality`, `${prefix}_color` and
     * `${prefix}_estimated_reliability` straight from the backend payload
     * (prefix is "forward" or "return"). Identifies the node that received
     * the measurements, shows RSSI only where available, and distinguishes
     * "observed, SNR unavailable" from "no observations in this direction".
     */
    function directionSection(options) {
        const prefix = options.prefix;
        const link = options.link;
        const senderName = options.senderName;
        const receiverName = options.receiverName;

        const snr = link[`${prefix}_avg_snr`];
        const rssi = link[`${prefix}_avg_rssi`];
        const reliability = link[`${prefix}_estimated_reliability`];
        const count = link[`${prefix}_count`];
        const quality = link[`${prefix}_quality`];
        const color = link[`${prefix}_color`];
        const observed = (count ?? 0) > 0 || hasValue(snr);

        const parts = [];
        const snrText = formatDb(snr);
        if (snrText !== null) {
            parts.push(snrText);
        } else if (observed) {
            // Direction was observed but produced no usable SNR sample.
            parts.push('SNR unavailable');
        }
        const rssiText = formatDbm(rssi);
        if (rssiText !== null) {
            parts.push(rssiText);
        }
        if ((count ?? 0) > 0) {
            parts.push(`${count} observation${count === 1 ? '' : 's'}`);
        }
        const reliabilityText = formatPercent(reliability);
        if (reliabilityText !== null) {
            parts.push(`est. ${reliabilityText}`);
        }

        return el('div', { className: 'rf-direction mb-1' },
            el('div', { className: 'small' },
                el('strong', null, `${senderName} → ${receiverName}`),
                textNode(' (received at '),
                el('em', null, receiverName),
                textNode('):')),
            el('div', { className: 'small' },
                textNode(parts.length > 0
                    ? ` ${parts.join(' · ')} `
                    : ' No observations in this direction '),
                qualityBadge(quality, color)));
    }

    /**
     * "Estimated reliability" summary row (0–100 model estimate for the
     * worst observed direction — signal headroom only, not measured
     * packet delivery).
     */
    function reliabilityRow(value) {
        if (!hasValue(value)) {
            return null;
        }
        return el('div', null,
            el('strong', null, 'Estimated reliability:'),
            textNode(` ${formatPercent(value)}`),
            textNode(' '),
            el('i', {
                className: 'bi bi-info-circle text-muted',
                title: 'Model estimate from fade margin (signal headroom only); not measured packet delivery.'
            }));
    }

    /**
     * Balance warning rendered from the backend `link_balance`
     * classification (or the legacy `is_bidirectional` flag). Distinguishes
     * "only one direction observed" from "observed, SNR unavailable" without
     * reclassifying anything client-side.
     */
    function balanceMessage(link) {
        const balance = link.link_balance;
        if (balance === 'balanced') {
            return el('div', { className: 'small text-success' },
                el('i', { className: 'bi bi-check-circle' }),
                textNode(' Balanced — both directions usable'));
        }
        if (balance === 'asymmetric_marginal') {
            return el('div', { className: 'small text-warning' },
                el('i', { className: 'bi bi-exclamation-triangle' }),
                textNode(' Asymmetric — one direction marginal'));
        }
        if (balance === 'marginal_both') {
            return el('div', { className: 'small text-danger' },
                el('i', { className: 'bi bi-exclamation-triangle' }),
                textNode(' Fragile — both directions marginal'));
        }
        if (balance === 'unidirectional') {
            return el('div', { className: 'small text-muted' },
                el('i', { className: 'bi bi-arrow-right' }),
                textNode(' Only one direction observed'));
        }
        if (balance === 'unknown') {
            const anyObserved = (link.forward_count ?? 0) > 0
                || (link.return_count ?? 0) > 0;
            return el('div', { className: 'small text-muted' },
                el('i', { className: 'bi bi-question-circle' }),
                textNode(anyObserved
                    ? ' Observed, but SNR unavailable for balance assessment'
                    : ' Directional balance unknown'));
        }
        if (link.is_bidirectional === false) {
            // Legacy payload without balance classification.
            return el('div', { className: 'small text-muted' },
                textNode('One direction observed — incomplete coverage'));
        }
        return null;
    }

    window.RFLinkUI = {
        hasValue,
        formatDb,
        formatDbm,
        formatPercent,
        qualityColor,
        qualityBadge,
        directionSection,
        reliabilityRow,
        balanceMessage
    };
})();
