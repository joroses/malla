/**
 * Lazy loader for heavyweight third-party chart libraries.
 *
 * Chart.js and Plotly are only needed on a handful of pages, yet loading
 * them eagerly blocks first paint (head) or DOMContentLoaded (body). Pages
 * call the matching loader once data is being fetched; the library then
 * downloads in parallel and resolves just before charts are rendered.
 *
 * Every artifact is self-hosted under /static/vendor and version-pinned, so
 * pages behave identically offline and no third-party CDN is contacted.
 */
(function () {
    'use strict';

    var SOURCES = {
        chartJs: {
            url: '/static/vendor/chart.js/4.4.0/chart.umd.min.js',
        },
        chartJsZoom: {
            // Must load after Chart.js: the plugin self-registers against the
            // global Chart object at execution time.
            url: '/static/vendor/chartjs-plugin-zoom/2.2.0/chartjs-plugin-zoom.min.js',
            after: 'chartJs',
        },
        plotly: {
            url: '/static/vendor/plotly/2.30.0/plotly-2.30.0.min.js',
        },
    };

    var pending = {};

    function inject(source) {
        return new Promise(function (resolve, reject) {
            var script = document.createElement('script');
            script.src = source.url;
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
