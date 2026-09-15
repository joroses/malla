// OpenFreeMap vector-tile basemap for Leaflet maps, rendered via a MapLibre GL WebGL overlay.
const OFM_LIGHT_STYLE = 'https://tiles.openfreemap.org/styles/liberty';
const OFM_DARK_STYLE = 'https://tiles.openfreemap.org/styles/dark';
const OFM_ATTRIBUTION = '© <a href="https://openfreemap.org" target="_blank" rel="noopener">OpenFreeMap</a> © <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a> © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';
const TERRAIN_DEM_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png';

// The hosted "liberty" style bundles a Natural Earth raster relief source
// ("ne2_shaded") whose hillshade PNGs account for the majority of basemap
// tile requests without adding road/label readability. The style is fetched
// here as JSON, stripped of that source, and the filtered object is handed
// to MapLibre, eliminating those tile requests entirely.
const OFM_EMPTY_STYLE = { version: 8, sources: {}, layers: [] };
const ofmStyleRequests = {}; // style URL -> Promise<filtered style object | null>
const ofmStyleObjects = {};  // style URL -> resolved filtered style object

function ofmStripHillshade(style) {
    if (style && Array.isArray(style.layers)) {
        style.layers = style.layers.filter(layer => layer && layer.source !== 'ne2_shaded');
    }
    if (style && style.sources) delete style.sources.ne2_shaded;
    return style;
}

function ofmLoadFilteredStyle(url) {
    if (!ofmStyleRequests[url]) {
        ofmStyleRequests[url] = fetch(url)
            .then(response => {
                if (!response.ok) throw new Error('style fetch failed: HTTP ' + response.status);
                return response.json();
            })
            .then(style => {
                ofmStyleObjects[url] = ofmStripHillshade(style);
                return ofmStyleObjects[url];
            })
            .catch(err => {
                console.warn('OpenFreeMap style fetch failed; falling back to the hosted style.', err);
                return null;
            });
    }
    return ofmStyleRequests[url];
}

function ofmIsDarkTheme() {
    const attr = document.documentElement.getAttribute('data-bs-theme');
    if (attr) return attr === 'dark';
    // The theme attribute is only applied at DOMContentLoaded; until then
    // resolve it the same way DarkModeToggle does (localStorage, then system).
    let preference = null;
    try { preference = localStorage.getItem('malla-theme-preference'); } catch (err) { /* storage unavailable */ }
    if (preference === 'dark') return true;
    if (preference === 'light') return false;
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
}

// Warm the filtered style fetch at script parse time so it overlaps with
// page/data loading instead of delaying the first basemap paint.
ofmLoadFilteredStyle(ofmIsDarkTheme() ? OFM_DARK_STYLE : OFM_LIGHT_STYLE);

// Recommended Leaflet map options for the GL adapter (see plugin README: maxBounds avoids
// the latitude-sync issue, minZoom avoids zoom-0 sync issues).
window.openFreeMapLeafletMapOptions = {
    maxBounds: [[180, -Infinity], [-180, Infinity]],
    maxBoundsViscosity: 1,
    minZoom: 1,
    maxZoom: 20,
};

function addTerrainLayers(glMap, contourSource) {
    try {
        glMap.addSource('terrarium-dem', { type: 'raster-dem', tiles: [TERRAIN_DEM_URL], tileSize: 256, encoding: 'terrarium', maxzoom: 15 });
        glMap.addLayer({ id: 'terrain-hillshade', type: 'hillshade', source: 'terrarium-dem',
            paint: { 'hillshade-exaggeration': 0.6, 'hillshade-illumination-direction': 315,
                     'hillshade-shadow-color': '#263238', 'hillshade-highlight-color': '#ffffff',
                     'hillshade-accent-color': '#607d8b' } });
        if (contourSource) {
            glMap.addSource('contours-dem', { type: 'raster-dem', tiles: [contourSource.sharedDemProtocolUrl], encoding: 'terrarium', tileSize: 256, maxzoom: 12 });
            glMap.addSource('contours', { type: 'vector', tiles: [contourSource.contourProtocolUrl({ thresholds: { 11: [200, 1000], 12: [100, 500], 13: [100, 500], 14: [50, 200], 15: [20, 100] } })], maxzoom: 15 });
            glMap.addLayer({ id: 'terrain-contours', type: 'line', source: 'contours', 'source-layer': 'contours',
                paint: { 'line-color': '#444444', 'line-opacity': 0.5, 'line-width': ['match', ['get', 'level'], 1, 1.2, 0.6] } });
        }
    } catch (err) {
        console.warn('OpenFreeMap terrain layers failed; continuing with plain basemap.', err);
    }
}

window.createOpenFreeMapOverlay = function (options = {}) {
    const styleUrl = ofmIsDarkTheme() ? OFM_DARK_STYLE : OFM_LIGHT_STYLE;

    // maplibre-contour must register its tile protocol BEFORE the GL map is created.
    let contourSource = null;
    if (options.terrain && window.mlcontour) {
        contourSource = new mlcontour.DemSource({ url: TERRAIN_DEM_URL, encoding: 'terrarium', maxzoom: 12, worker: true });
        contourSource.setupMaplibre(maplibregl);
    }

    if (typeof L.maplibreGL !== 'function') {
        console.warn('maplibre-gl-leaflet failed to load; map basemap disabled.');
        return null;
    }

    const cached = ofmStyleObjects[styleUrl];
    const overlay = L.maplibreGL({
        // Render the filtered style immediately when it is already fetched;
        // otherwise bootstrap from an empty style (zero tile requests) and
        // swap the real style in once the fetch resolves.
        style: cached ? structuredClone(cached) : OFM_EMPTY_STYLE,
        attributionControl: { customAttribution: OFM_ATTRIBUTION },
    });

    // The inner GL map is created in Leaflet's onAdd, i.e. only after
    // overlay.addTo(map): hook style swap / terrain setup there.
    overlay.once('add', () => {
        if (cached) {
            if (options.terrain) {
                const glMap = overlay.getMaplibreMap();
                if (glMap) glMap.once('style.load', () => addTerrainLayers(glMap, contourSource));
            }
            return;
        }
        ofmLoadFilteredStyle(styleUrl).then(filtered => {
            // getMaplibreMap() returns null once the overlay has been
            // removed (e.g. a theme switch), making the swap a safe no-op.
            const glMap = overlay.getMaplibreMap();
            if (!glMap) return;
            glMap.setStyle(filtered ? structuredClone(filtered) : styleUrl, { diff: false });
            if (options.terrain) {
                // Terrain layers must attach to the final style, not the
                // empty bootstrap style that setStyle is about to replace.
                glMap.once('style.load', () => addTerrainLayers(glMap, contourSource));
            }
        });
    });
    return overlay;
};
