/**
 * Lazy loader for heavyweight third-party chart libraries.
 *
 * Chart.js and Plotly are only needed on a handful of pages, yet loading
 * them eagerly blocks first paint (head) or DOMContentLoaded (body). Pages
 * call the matching loader once data is being fetched; the library then
 * downloads in parallel and resolves just before charts are rendered.
 *
 * Every URL is version-pinned and SRI-hashed, so a tampered or drifting
 * CDN artifact fails to execute instead of silently changing behavior.
 */
(function () {
    'use strict';

    var SOURCES = {
        chartJs: {
            url: 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js',
            integrity: 'sha384-e6nUZLBkQ86NJ6TVVKAeSaK8jWa3NhkYWZFomE39AvDbQWeie9PlQqM3pmYW5d1g',
        },
        chartJsZoom: {
            // Must load after Chart.js: the plugin self-registers against the
            // global Chart object at execution time.
            url: 'https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js',
            integrity: 'sha384-dwwI6ICEN/0ZQlS5owhUa/6ZzvwUPmjH45bFVCAcjgjTulbHJvlE+TGU3g1k0N3R',
            after: 'chartJs',
        },
        plotly: {
            url: 'https://cdn.plot.ly/plotly-2.30.0.min.js',
            integrity: 'sha384-H7GB7Kme/VbPI/0S4LNq7OixFNVRgRGE8kyqTntBuiXle1KBm8KWLQh/Ah6bXCYW',
        },
    };

    var pending = {};

    function inject(source) {
        return new Promise(function (resolve, reject) {
            var script = document.createElement('script');
            script.src = source.url;
            script.integrity = source.integrity;
            script.crossOrigin = 'anonymous';
            script.onload = function () { resolve(); };
            script.onerror = function () {
                reject(new Error('Failed to load ' + source.url));
            };
            document.head.appendChild(script);
        });
    }

    function load(name) {
        if (!pending[name]) {
            var chain = Promise.resolve();
            var source = SOURCES[name];
            if (source.after) {
                chain = load(source.after);
            }
            pending[name] = chain.then(function () { return inject(source); });
            // Allow a retry after a failed attempt.
            pending[name].catch(function () { delete pending[name]; });
        }
        return pending[name];
    }

    window.MallaVendor = {
        /** Chart.js 4.x global (window.Chart). */
        chartJs: function () { return load('chartJs'); },
        /** Chart.js plus the zoom plugin, in registration order. */
        chartJsZoom: function () { return load('chartJsZoom'); },
        /** Plotly 2.x global (window.Plotly). */
        plotly: function () { return load('plotly'); },
    };
})();
