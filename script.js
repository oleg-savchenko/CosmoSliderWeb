// --- Helper to read URL parameters ---
function getQueryParam(name) {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get(name);
}

// --- Helper to inject Umami ---
function loadUmami() {
  if (window.__umamiLoaded) return; // prevent duplicate loads
  window.__umamiLoaded = true;

  const s = document.createElement("script");
  s.defer = true;
  s.src = "https://cloud.umami.is/script.js";
  s.setAttribute("data-website-id", "81b626e1-16f6-4963-b3b8-4f9d34cd7fb9");
  document.head.appendChild(s);
}

// --- Lazy activation when tracking=false ---
function enableUmamiOnInteraction() {
  const interactionEvents = ["click", "scroll", "keydown", "mousemove", "touchstart"];

  const triggerLoad = () => {
    loadUmami();
    interactionEvents.forEach(evt => window.removeEventListener(evt, triggerLoad));
  };

  interactionEvents.forEach(evt => {
    window.addEventListener(evt, triggerLoad, { once: true, passive: true });
  });
}

// --- On DOM ready ---
window.addEventListener("DOMContentLoaded", () => {
  // Handle dark mode
  const darkParam = getQueryParam("dark");
  if (darkParam === "1" || darkParam === "true" || darkParam === "yes") {
    document.body.classList.add("dark-mode");
  }

  // Handle tracking
  const trackingParam = getQueryParam("tracking");
  const trackingDisabled = (trackingParam === "0" || trackingParam === "false" || trackingParam === "no");

  if (trackingDisabled) {
    // Only load Umami after user interaction
    enableUmamiOnInteraction();
  } else {
    // Load Umami immediately
    loadUmami();
  }
});

// --- Dark mode toggle button ---
document.getElementById("toggleDarkMode").addEventListener("click", () => {
    document.body.classList.toggle("dark-mode");
    updateAxesAndChart();
});

const CMB_MODES = ["TT", "TE", "EE", "PP"];
const MATTER_MODES = ["PK_LIN", "PK_NONLIN"];
const SPECTRUM_STORAGE_KEY = "cosmoslider_selected_spectrum";
const MATTER_Y_MIN = 1e-2;
const MATTER_Y_MAX = 1e6;
const MATTER_Y_AXIS_MAX = MATTER_Y_MAX * 1.0000001;
const MATTER_COMPILATION_DATA_URL = 'mpk_compilation_data.json';

const models = {
    cmb: null,
    matterLinear: null,
    matterNonlinear: null,
};
let matterKGrid = [];
let matterCompilationData = null;

const sliders = Array.from(document.querySelectorAll(".slider"));
const sliderById = {};
for (const slider of sliders) {
    sliderById[slider.id] = slider;
}

const CMB_SLIDER_SPECS = {
    slider1: { min: 0.014, max: 0.03, step: 0.0001, value: 0.0223 },
    slider2: { min: 0.0, max: 0.3, step: 0.001, value: 0.119 },
    slider3: { min: 50, max: 120, step: 0.1, value: 67.7 },
    slider4: { min: 1, max: 5, step: 0.01, value: 3.04 },
    slider5: { min: 0.5, max: 1.5, step: 0.001, value: 0.965 },
    slider6: { min: 0.0, max: 0.2, step: 0.001, value: 0.054 },
};

const MATTER_SLIDER_SPECS = {
    slider1: { min: 0.01875, max: 0.02625, step: 0.00001, value: 0.0223 },
    slider2: { min: 0.05, max: 0.255, step: 0.001, value: 0.119 },
    slider3: { min: 64, max: 82, step: 0.1, value: 67.7 },
    slider4: { min: 1.61, max: 3.91, step: 0.01, value: 3.04 },
    slider5: { min: 0.84, max: 1.10, step: 0.001, value: 0.965 },
    slider7: { min: 0, max: 5, step: 0.01, value: 0 },
    slider8: { min: 2, max: 4, step: 0.01, value: 3 },
    slider9: { min: 0.5, max: 1.0, step: 0.001, value: 0.75 },
};

function isMatterMode(mode = selectedOption.value) {
    return MATTER_MODES.includes(mode);
}

function isCmbMode(mode = selectedOption.value) {
    return CMB_MODES.includes(mode);
}

let mathJaxTypesetPromise = Promise.resolve();

function typesetMathLabels() {
    if (!window.MathJax || !MathJax.typesetPromise) {
        return;
    }

    const elements = Array.from(document.querySelectorAll(".math-slider-label, #outputChart .highcharts-container"));
    mathJaxTypesetPromise = mathJaxTypesetPromise
        .then(() => MathJax.typesetPromise(elements))
        .then(() => {
            alignChartMathLabels();
            requestAnimationFrame(alignChartMathLabels);
        })
        .catch(error => console.error("MathJax typesetting failed", error));
}

function alignChartMathLabels() {
    const chart = Highcharts.charts.find(Boolean);
    const xAxes = (chart && chart.xAxis) || [];
    const yAxis = chart && chart.yAxis && chart.yAxis[0];

    if (!chart || !xAxes[0] || !xAxes[0].ticks) {
        return;
    }

    const containerRect = chart.container.getBoundingClientRect();

    function setTickLength(tick, length) {
        const markElement = tick.mark && tick.mark.element;

        if (!markElement) {
            return;
        }

        const pathValues = (markElement.getAttribute("d") || "").match(/-?\d+(?:\.\d+)?/g);

        if (!pathValues || pathValues.length < 4) {
            return;
        }

        const x1 = parseFloat(pathValues[0]);
        const y1 = parseFloat(pathValues[1]);
        const x2 = parseFloat(pathValues[2]);
        const y2 = parseFloat(pathValues[3]);
        const horizontalTick = Math.abs(x2 - x1) > Math.abs(y2 - y1);
        const direction = horizontalTick ? Math.sign(x2 - x1) || 1 : Math.sign(y2 - y1) || 1;
        const endX = horizontalTick ? x1 + direction * length : x1;
        const endY = horizontalTick ? y1 : y1 + direction * length;
        markElement.setAttribute("d", `M ${x1} ${y1} L ${endX} ${endY}`);
    }

    function allowVisibleOverflow(element) {
        let current = element;

        while (current && current !== chart.container) {
            if (current.style) {
                current.style.overflow = "visible";
                current.style.textOverflow = "clip";
            }

            current = current.parentElement;
        }

        for (const mathElement of element.querySelectorAll("mjx-container, mjx-container svg")) {
            mathElement.style.overflow = "visible";
        }
    }

    function getMathRect(element) {
        const mathElement = element.querySelector("mjx-container") || element;
        return mathElement.getBoundingClientRect();
    }

    function adjustTickLabels(axis) {
        if (!axis || !axis.ticks) {
            return;
        }

        const isXAxis = Boolean(axis.isXAxis);
        const adjustTickLengths = isXAxis || axis.options.type === "logarithmic";

        for (const tick of Object.values(axis.ticks)) {
            const labelElement = tick.label && tick.label.element;
            const hasVisibleLabel = Boolean(labelElement && labelElement.textContent.trim());

            if (adjustTickLengths) {
                setTickLength(tick, hasVisibleLabel ? 15 : 8);
            }

            if (!labelElement || !labelElement.querySelector || !labelElement.querySelector("mjx-container")) {
                continue;
            }

            allowVisibleOverflow(labelElement);
            labelElement.style.transform = "";

            const labelRect = getMathRect(labelElement);
            const tickValue = axis.options.type === "logarithmic" ? Math.pow(10, tick.pos) : tick.pos;
            const tickCenter = axis.toPixels(tickValue);

            if (isXAxis) {
                const labelCenter = labelRect.left - containerRect.left + labelRect.width / 2;
                const horizontalOffset = tickCenter - labelCenter;
                let verticalOffset = 0;

                if (tick.mark && tick.mark.element) {
                    const tickRect = tick.mark.element.getBoundingClientRect();
                    const targetGap = 10;

                    if (axis.opposite) {
                        const currentGap = tickRect.top - labelRect.bottom;
                        verticalOffset = currentGap - targetGap;
                    } else {
                        const currentGap = labelRect.top - tickRect.bottom;
                        verticalOffset = targetGap - currentGap;
                    }
                }

                labelElement.style.transform = `translate(${horizontalOffset}px, ${verticalOffset}px)`;
            } else if (axis.options.type === "logarithmic") {
                const labelCenter = labelRect.top - containerRect.top + labelRect.height / 2;
                const offset = tickCenter - labelCenter;
                labelElement.style.transform = `translateY(${offset}px)`;
            }
        }
    }

    function adjustAxisTitle(axis) {
        const titleElement = axis && axis.axisTitle && axis.axisTitle.element;

        if (!axis || !titleElement || !titleElement.querySelector || !titleElement.querySelector("mjx-container")) {
            return;
        }

        allowVisibleOverflow(titleElement);
        titleElement.style.transform = "";

        const titleRect = getMathRect(titleElement);

        if (axis.isXAxis) {
            const titleCenter = titleRect.left - containerRect.left + titleRect.width / 2;
            const axisCenter = chart.plotLeft + chart.plotWidth / 2;
            const horizontalOffset = axisCenter - titleCenter;

            const labeledTickRects = Object.values(axis.ticks)
                .map(tick => tick.label && tick.label.element)
                .filter(labelElement => labelElement && labelElement.textContent.trim() && labelElement.querySelector("mjx-container"))
                .map(getMathRect);
            const titleGap = 12;
            let verticalOffset = 0;

            if (labeledTickRects.length) {
                if (axis.opposite) {
                    const topmostLabelTop = Math.min(...labeledTickRects.map(rect => rect.top));
                    const targetTitleBottom = topmostLabelTop - titleGap;
                    verticalOffset = targetTitleBottom - titleRect.bottom;
                } else {
                    const bottommostLabelBottom = Math.max(...labeledTickRects.map(rect => rect.bottom));
                    const targetTitleTop = bottommostLabelBottom + titleGap;
                    verticalOffset = targetTitleTop - titleRect.top;
                }
            }

            titleElement.style.transform = `translate(${horizontalOffset}px, ${verticalOffset}px)`;
        } else if (axis === yAxis) {
            const titleCenter = titleRect.top - containerRect.top + titleRect.height / 2;
            const axisCenter = chart.plotTop + chart.plotHeight / 2;
            const verticalOffset = axisCenter - titleCenter;
            const labeledTickRects = Object.values(axis.ticks)
                .map(tick => tick.label && tick.label.element)
                .filter(labelElement => labelElement && labelElement.textContent.trim() && labelElement.querySelector("mjx-container"))
                .map(getMathRect);
            const leftmostTickLabel = Math.min(...labeledTickRects.map(rect => rect.left));
            const titleGap = axis.options.type === "logarithmic" ? 12 : 18;
            const targetTitleRight = Number.isFinite(leftmostTickLabel)
                ? leftmostTickLabel - titleGap
                : containerRect.left + 58;
            const minimumTitleLeft = 8;
            const unclampedHorizontalOffset = targetTitleRight - titleRect.right;
            const horizontalOffset = Math.max(unclampedHorizontalOffset, minimumTitleLeft - titleRect.left);
            titleElement.style.transform = `translate(${-verticalOffset}px, ${horizontalOffset}px)`;
        }
    }

    for (const axis of xAxes) {
        adjustTickLabels(axis);
        adjustAxisTitle(axis);
    }
    adjustTickLabels(yAxis);
    adjustAxisTitle(yAxis);
}

function latexInline(value) {
    return `\\(${value}\\)`;
}

function getCmbYAxisTitle(mode) {
    if (mode === "PP") {
        return latexInline('\\frac{\\ell^2(\\ell+1)^2}{2\\pi}C_\\ell^{\\phi\\phi}\\;[\\times 10^7]');
    }

    return latexInline(`\\frac{\\ell(\\ell+1)}{2\\pi}C_\\ell^{\\mathrm{${mode}}}\\;[\\mu\\mathrm{K}^2]`);
}

function getMatterYAxisTitle(mode) {
    const symbol = mode === "PK_NONLIN" ? 'P_{\\mathrm{m}}^{\\mathrm{non\\text{-}lin}}' : 'P_{\\mathrm{m}}';
    return latexInline(`${symbol}\\;(k)\\;[(\\mathrm{Mpc}/h)^3]`);
}

function formatLatexNumber(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return value.toString();
    }

    if (Number.isInteger(numericValue)) {
        return numericValue.toString();
    }

    return numericValue.toString();
}

function formatLatexLogNumber(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue) || numericValue <= 0) {
        return formatLatexNumber(value);
    }

    const exponent = Math.round(Math.log10(numericValue));
    const isPowerOfTen = Math.abs(Math.log10(numericValue) - exponent) < 1e-8;

    if (isPowerOfTen) {
        if (exponent === 0) {
            return "1";
        }

        if (exponent === 1) {
            return "10";
        }

        return `10^{${exponent}}`;
    }

    return formatLatexNumber(numericValue);
}

function formatLatexLogTickLabel(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue) || numericValue <= 0) {
        return "";
    }

    const exponent = Math.round(Math.log10(numericValue));
    const isPowerOfTen = Math.abs(Math.log10(numericValue) - exponent) < 1e-8;

    return isPowerOfTen ? latexInline(formatLatexLogNumber(numericValue)) : "";
}

function formatAngleTickLabel(degrees) {
    return latexInline(`${degrees}^{\\circ}`);
}

// Angular scale (top axis for CMB plots) shares the same nonlinear log-then-linear
// transform as the bottom multipole axis (scaleGraphPoint), via theta[deg] = 180 / ell.
// Candidates outside the plotted ell range (i.e. transformed position outside [0,1])
// are dropped rather than clamped, so the tick set adapts to whatever ell range is in use.
function generateAngleTicks(candidateDegrees) {
    const tickPositions = [];
    const tickLabels = [];
    const epsilon = 1e-6;

    for (const degrees of candidateDegrees) {
        const ell = 180 / degrees;
        const position = scaleGraphPoint(ell);

        if (position >= -epsilon && position <= 1 + epsilon) {
            tickPositions.push(position);
            tickLabels.push(formatAngleTickLabel(degrees));
        }
    }

    return { tickPositions, tickLabels };
}

// Spatial scale (top axis for matter power spectrum plots) is the wavelength
// lambda = 2*pi/k [Mpc/h]. Build ticks in lambda-space first, then convert each
// tick back to the linked k-axis position so major and minor ticks stay aligned.
function generateScaleAxisTicks(kMin, kMax) {
    const tickPositions = [];
    const tickLabels = [];
    const epsilon = 1e-9;
    const logScaleMin = Math.log10((2 * Math.PI) / kMax);
    const logScaleMax = Math.log10((2 * Math.PI) / kMin);
    const logKMin = Math.log10(kMin);
    const logKMax = Math.log10(kMax);

    for (let exponent = Math.floor(logScaleMin); exponent <= Math.ceil(logScaleMax); exponent++) {
        for (let mantissa = 1; mantissa <= 9; mantissa++) {
            const logScale = exponent + Math.log10(mantissa);
            const logK = Math.log10(2 * Math.PI) - logScale;

            if (logK < logKMin - epsilon || logK > logKMax + epsilon) {
                continue;
            }

            tickPositions.push(logK);
            tickLabels.push(mantissa === 1 ? latexInline(formatLatexLogNumber(10 ** exponent)) : "");
        }
    }

    return mergeSortedUniqueWithLabels(tickPositions, tickLabels);
}

function findClosestIndex(sortedLogPositions, targetLogPosition, tolerance) {
    for (let i = 0; i < sortedLogPositions.length; i++) {
        if (Math.abs(sortedLogPositions[i] - targetLogPosition) < tolerance) {
            return i;
        }
    }
    return -1;
}

function mergeSortedUniqueWithLabels(values, labels) {
    return values
        .map((value, index) => ({ value, label: labels[index] }))
        .sort((a, b) => a.value - b.value)
        .filter((item, index, items) => index === 0 || Math.abs(item.value - items[index - 1].value) > 1e-10)
        .reduce((result, item) => {
            result.tickPositions.push(item.value);
            result.tickLabels.push(item.label);
            return result;
        }, { tickPositions: [], tickLabels: [] });
}

function getGridLineColor(isDarkMode) {
    return isDarkMode ? 'rgba(255, 255, 255, 0.07)' : 'rgba(0, 0, 0, 0.06)';
}

// A slight, non-distracting grid through only the labeled (major) ticks of an axis,
// via plotLines rather than gridLineWidth, since tickPositions here are usually a
// merged major+minor set and gridLineWidth would draw a line at every one of them.
// Note: plotLines always take the axis's REAL value, even on a logarithmic axis -
// unlike tickPositions in this codebase's convention, which are in log10-space for
// logarithmic axes (see generateLogTicks). Callers on a logarithmic axis must pass
// real values (e.g. via realLogPositions) here, not the log10-space tick positions.
function buildGridPlotLines(positions, color) {
    return positions.map(value => ({
        value,
        color,
        width: 1,
        dashStyle: 'ShortDot',
        zIndex: 0,
    }));
}

function realLogPositions(logPositions) {
    return logPositions.map(position => Math.pow(10, position));
}

function generateLogTicks(logMin, logMax) {
    const majorTicks = [];
    const minorTicks = [];
    const epsilon = 1e-10;

    for (let exponent = Math.ceil(logMin - epsilon); exponent <= Math.floor(logMax + epsilon); exponent++) {
        majorTicks.push(exponent);
    }

    for (let exponent = Math.floor(logMin); exponent <= Math.ceil(logMax); exponent++) {
        for (let mantissa = 2; mantissa <= 9; mantissa++) {
            const tick = exponent + Math.log10(mantissa);

            if (tick > logMin + epsilon && tick < logMax - epsilon) {
                minorTicks.push(tick);
            }
        }
    }

    return mergeSortedUnique(majorTicks, minorTicks);
}

function mergeSortedUnique(...arrays) {
    return arrays
        .flat()
        .sort((a, b) => a - b)
        .filter((value, index, values) => index === 0 || Math.abs(value - values[index - 1]) > 1e-10);
}

function updateSliderFill(slider) {
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const currentVal = parseFloat(slider.value);
    const percent = max === min ? 0 : ((currentVal - min) / (max - min)) * 100;
    slider.style.backgroundSize = `${percent}% 100%`;
}

function updateSliderReadout(slider) {
    const valueElement = document.getElementById(`${slider.id}-value`);
    if (valueElement) {
        valueElement.innerText = parseFloat(slider.value).toString();
    }
}

function setSliderSpec(slider, spec, resetValue = false) {
    const currentValue = parseFloat(slider.value);
    slider.min = spec.min;
    slider.max = spec.max;
    slider.step = spec.step;

    if (resetValue) {
        slider.value = spec.value;
    } else {
        const value = Number.isFinite(currentValue) ? currentValue : spec.value;
        slider.value = Math.min(Math.max(value, spec.min), spec.max);
    }

    updateSliderFill(slider);
    updateSliderReadout(slider);
}

function setSliderVisible(sliderId, visible) {
    const slider = sliderById[sliderId];
    if (slider) {
        slider.parentElement.style.display = visible ? "" : "none";
    }
}

function setShowDataEnabled(enabled) {
    const checkbox = document.getElementById("toggle");
    const wrapper = checkbox.closest(".toggle-checkbox-wrapper");
    checkbox.disabled = !enabled;
    wrapper.classList.toggle("disabled", !enabled);

    if (!enabled) {
        checkbox.checked = false;
        checkboxChecked = false;
        data = null;
    }
}

function syncModeControls(resetValues = false) {
    const matterMode = isMatterMode();
    const nonlinearMatterMode = selectedOption.value === "PK_NONLIN";
    const specs = matterMode ? MATTER_SLIDER_SPECS : CMB_SLIDER_SPECS;

    for (const [sliderId, spec] of Object.entries(specs)) {
        setSliderSpec(sliderById[sliderId], spec, resetValues);
    }

    setSliderVisible("slider6", !matterMode);
    setSliderVisible("slider7", matterMode);
    setSliderVisible("slider8", nonlinearMatterMode);
    setSliderVisible("slider9", nonlinearMatterMode);
    setShowDataEnabled(true);
    typesetMathLabels();
}

for (const slider of sliders) {
    slider.addEventListener("input", () => {
        updateSliderFill(slider);
        updateDataAndChart();
    });
}





function loadData(name) {
    const request = new XMLHttpRequest();
    request.open('GET', name, false); // false makes the request synchronous
    request.send(null);

    if (request.status === 200 || (request.status === 0 && request.responseText)) {
        return JSON.parse(request.responseText);
    } else {
        console.error(`Failed to load ${name}`);
        return null;
    }
}

function localFileFetch(url) {
    return new Promise((resolve, reject) => {
        const requestUrl = typeof url === 'string' ? url : url.url;
        const request = new XMLHttpRequest();
        request.open('GET', requestUrl, true);
        request.responseType = 'arraybuffer';

        request.onload = () => {
            if (request.status === 200 || (request.status === 0 && request.response)) {
                const headers = new Headers();
                if (requestUrl.toLowerCase().split('?')[0].endsWith('.json')) {
                    headers.set('Content-Type', 'application/json');
                } else {
                    headers.set('Content-Type', 'application/octet-stream');
                }
                resolve(new Response(request.response, {
                    status: 200,
                    statusText: 'OK',
                    headers,
                }));
            } else {
                reject(new Error(`Failed to load ${requestUrl}`));
            }
        };

        request.onerror = () => reject(new Error(`Failed to load ${requestUrl}`));
        request.send(null);
    });
}

function loadGraphModel(modelUrl) {
    if (window.location.protocol === 'file:') {
        return tf.loadGraphModel(modelUrl, { fetchFunc: localFileFetch });
    }

    return tf.loadGraphModel(modelUrl);
}

const tt_data = loadData('tt_data.json');
const te_data = loadData('te_data.json');
const ee_data = loadData('ee_data.json');
const pp_data = loadData('pp_data.json');
const mpk_compilation_data = loadData(MATTER_COMPILATION_DATA_URL);


var data = null;
var checkboxChecked = false;



function resetSlidersToBestfit() {
    syncModeControls(true);
}

function resetGraphToBestfit() {
    resetSlidersToBestfit();
    updateDataAndChart();
}


const selectedOption = { value: "TT" };

function getCmbDataForMode(mode) {
    if (mode === "TT") {
        return tt_data;
    } else if (mode === "TE") {
        return te_data;
    } else if (mode === "EE") {
        return ee_data;
    } else if (mode === "PP") {
        return pp_data;
    }

    return null;
}

function syncCurrentData() {
    if (!checkboxChecked) {
        data = null;
    } else if (isCmbMode()) {
        data = getCmbDataForMode(selectedOption.value);
    } else if (isMatterMode()) {
        data = matterCompilationData;
    } else {
        data = null;
    }
}

// Function to handle dropdown selection
function handleSelection(event) {
    // Set the selected value in the object
    selectedOption.value = event.target.value;
    localStorage.setItem(SPECTRUM_STORAGE_KEY, selectedOption.value);
    syncModeControls(false);
    syncCurrentData();
    updateAxes();
    updateDataAndChart();
}

function restoreSavedSpectrum() {
    const saved = localStorage.getItem(SPECTRUM_STORAGE_KEY);
    if (!saved || !(CMB_MODES.includes(saved) || MATTER_MODES.includes(saved))) {
        return;
    }
    selectedOption.value = saved;
    document.getElementById("dropdown").value = saved;
}

function showData(checkboxElem) {
    checkboxChecked = checkboxElem.checked;
    syncCurrentData();
    updateAxes();
    updateDataAndChart();
}



window.addEventListener('load', async () => {
    matterKGrid = loadData('web_model/mpk_modes.json') || [];
    matterCompilationData = mpk_compilation_data;
    restoreSavedSpectrum();
    syncModeControls(true);
    syncCurrentData();

    models.cmb = await loadGraphModel('web_model/model.json');
    models.matterLinear = await loadGraphModel('web_model/mpk_lin/model.json');
    models.matterNonlinear = await loadGraphModel('web_model/mpk_nonlin/model.json');

    document.body.addEventListener("click", updateAxesAndChart);

    window.addEventListener("resize", updateAxesAndChart);

    // Initial chart display
    updateAxes();
    updateDataAndChart();
});




// initialize dictionaries. One contains axis plot configurations, and the other contains the data to be plotted
const axisConfig = {
    chart: {
	type: 'spline',
	backgroundColor: '#FFFFFF',
	marginLeft: 80,
	marginRight: 30,
	marginTop: 15,
	plotBorderWidth: 2,
	plotBorderColor: 'black',
    },
    title: {
		text: ''
    },
    xAxis: {
	type: 'linear',
	tickPositions: [],
	labels: {
	    formatter: function() {
		return '';
	    },
	    style: {
		color: 'black',
		fontSize: '20px'
	    },
	    rotation: 0,
	    overflow: 'allow',
	},
	title: {
	    text: latexInline('\\mathrm{Multipole},\\ \\ell'),
	    useHTML: true,
	    align: 'middle',
	},
	lineColor: 'black',
	lineWidth: 2,
	tickWidth: 2,
	tickLength: 10,
	tickColor: 'black',
	min: 0,
	max: 1,
	},
    yAxis: {
	tickPositions: [0, 2000, 4000, 6000, 8000],
	labels: {
	    formatter: function() {
		return ['0', '2000', '4000', '6000', '8000'].includes(this.value.toString()) ? this.value : '';
	    },
	    style: {
		color: 'black',
		fontSize: '20px'
	    },
	    rotation: 270,
	},
	title: {
	    text: getCmbYAxisTitle("TT"),
	    useHTML: true,
	    rotation: 270,
	    align: 'middle',
	    margin: 0,
	    style: {
		color: 'black',
		fontSize: '20px'
	    },
	},
	gridLineWidth: 0,
	lineWidth: 2,
	lineColor: 'black',
	tickWidth: 2,
	tickLength: 10,
	tickColor: 'black',
	alignTicks: false,
	endOnTick: false,
	startOnTick: false,
	min: 0,
	max: 8400,
    },
    series: [{
	data: [],
	lineWidth: 6,
	marker: {
	    enabled: false
	},
	color: 'rgb(38, 135, 242)'
    },{
	type: 'scatter',
	data: [],
	opacity: 0.6,
	marker: {
	    symbol: 'circle',
	    radius: 4,
	    fillColor: 'limeGreen',
	}
    },{
	type: 'errorbar',
	data: [],
	linkedTo: ':previous',
	stemWidth: 3,
	stemColor: 'limeGreen',
	whiskerLength: 0,
	opacity: 0.6,
	pointRange: 0,
	pointWidth: 3,
    }],
    legend: {
	enabled: false
    },
    tooltip: {
	enabled: false
    },
    credits: {
	enabled: false
    }
};

function createModelSeries(lineWidth = 6) {
    return {
        data: [],
        lineWidth,
        marker: {
            enabled: false
        },
        color: 'rgb(38, 135, 242)',
        name: 'Model',
        showInLegend: false
    };
}

function createCmbDataSeries(pointSize = 4) {
    return {
        type: 'scatter',
        data: [],
        opacity: 0.6,
        showInLegend: false,
        marker: {
            symbol: 'circle',
            radius: pointSize,
            fillColor: 'limeGreen',
        }
    };
}

function createCmbErrorSeries(stemWidth = 3) {
    return {
        type: 'errorbar',
        data: [],
        linkedTo: ':previous',
        showInLegend: false,
        stemWidth,
        stemColor: 'limeGreen',
        whiskerLength: 0,
        opacity: 0.6,
        pointRange: 0,
        pointWidth: 3,
    };
}

function resetCmbSeries(lineWidth, pointSize, stemWidth) {
    axisConfig.series = [
        createModelSeries(lineWidth),
        createCmbDataSeries(pointSize),
        createCmbErrorSeries(stemWidth),
    ];
}

function resetMatterSeries(lineWidth) {
    axisConfig.series = [createModelSeries(lineWidth)];
}


const logscale = 200.0;
const transition = 0.3;
const l_min = 2.0;

function generateXTicks(fewTicks = false) {
    let majorTicks = [];
    let minorTicks = [];
    let logOrder = 1;
    var linMajor = 500;
    var linMinor = 100;
    var minorTicksLogFactor = 1;
    if (fewTicks) {
	linMajor = 1000;
	linMinor = 200;
	minorTicksLogFactor = 2;
	}
    
    for (let i = 2; i <= 2500; i++) {
        if (i < logscale) {
            if (Number.isInteger(Math.log10(i))) {
                majorTicks.push(logScale(i));
                logOrder *= 10;
            } else if (i/minorTicksLogFactor % logOrder === 0) {
                minorTicks.push(logScale(i));
            }
        } else {
            if (i % linMajor === 0) {
                majorTicks.push(linScale(i));
            } else if (i % linMinor === 0) {
                minorTicks.push(linScale(i));
            }
        }
    }
    
    return { majorTicks, minorTicks };
}

function logScale(x) {
    const logx = Math.log10(x);
    const logxnew = (logx / Math.log10(logscale)) * transition;
    
    const loglmin = (Math.log10(l_min) / Math.log10(logscale)) * transition;
    const logscaleVal = (Math.log10(logscale) / Math.log10(logscale)) * transition;
    const logdiff = logscaleVal - loglmin;
    
    return ((logxnew - loglmin) * transition / logdiff);
}

function linScale(x) {
    const linxnew = (x / (2500 - logscale)) - (logscale / (2500 - logscale));
    return ((linxnew * (1 - transition)) + transition);
}

function scaleGraphPoint(xValue) {
    if (xValue <= logscale) {
        return logScale(xValue);
    } else {
        return linScale(xValue);
    }
}

// Function to merge, sort arrays based on values, and synchronize labels
function mergeSortAndSyncArrays(arr1, labels1, arr2, labels2) {
    // Combine arrays and labels into array of objects
    let combinedData = [];
    
    // Combine data from arr1 with labels from labels1
    for (let i = 0; i < arr1.length; i++) {
        combinedData.push({ value: arr1[i], label: labels1[i] });
    }
    
    // Combine data from arr2 with labels from labels2
    for (let i = 0; i < arr2.length; i++) {
        combinedData.push({ value: arr2[i], label: labels2[i] });
    }

    // Sort combined data based on value
    combinedData.sort((a, b) => a.value - b.value);

    // Extract sorted arrays and labels from sorted combined data
    let sortedValues = [];
    let sortedLabels = [];

    for (let item of combinedData) {
        sortedValues.push(item.value);
        sortedLabels.push(item.label);
    }

    return {
        sortedValues: sortedValues,//.map(v => ({value: v})),
        sortedLabels: sortedLabels
    };
}



async function updateAxesAndChart() {
    updateAxes();
    await updateDataAndChart();
}


function updateMatterAxes() {
    const isDarkMode = document.body.classList.contains('dark-mode');
    const axisColor = isDarkMode ? 'white' : 'black';
    const outputChart = document.getElementById('outputChart');
    const compactChart = outputChart.offsetWidth <= 500 || outputChart.offsetHeight < 400;
    const fontSize = compactChart ? '14px' : '20px';
    const lineWidth = compactChart ? 3 : 6;
    const marginLeft = compactChart ? 90 : 115;
    const marginRight = compactChart ? 20 : 30;
    const marginBottom = compactChart ? 100 : 115;
    const displayedMatterKGrid = matterKGrid.length ? getDisplayedMatterKGrid() : [];
    const kMin = displayedMatterKGrid.length ? displayedMatterKGrid[0] : 1e-5;
    const kDataMax = displayedMatterKGrid.length ? displayedMatterKGrid[displayedMatterKGrid.length - 1] : 10;
    const kMax = Math.max(10, kDataMax);
    const xLogTicks = generateLogTicks(Math.log10(kMin), Math.log10(kMax));
    const yLogTicks = generateLogTicks(Math.log10(MATTER_Y_MIN), Math.log10(MATTER_Y_AXIS_MAX));

    const { tickPositions: scaleTickPositions, tickLabels: scaleTickLabels } =
        generateScaleAxisTicks(kMin, kMax);
    const gridColor = getGridLineColor(isDarkMode);
    const xGridPlotLines = buildGridPlotLines(realLogPositions(xLogTicks.filter(Number.isInteger)), gridColor);
    const yGridPlotLines = buildGridPlotLines(realLogPositions(yLogTicks.filter(Number.isInteger)), gridColor);

    axisConfig.chart = {
        type: 'spline',
        backgroundColor: isDarkMode ? '#000000' : '#FFFFFF',
        animation: false,
        marginLeft: marginLeft,
        marginRight: marginRight,
        marginBottom: marginBottom,
        marginTop: compactChart ? 90 : 110,
        plotBorderWidth: 2,
        plotBorderColor: axisColor,
    };
    axisConfig.title = {
        text: '',
    };
    axisConfig.xAxis = [{
        type: 'logarithmic',
        tickPositions: xLogTicks,
        labels: {
            useHTML: true,
            formatter: function() {
                return formatLatexLogTickLabel(this.value);
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        title: {
            text: latexInline('\\mathrm{Wavevector}\\,k\\;[h/\\mathrm{Mpc}]'),
            useHTML: true,
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        lineColor: axisColor,
        lineWidth: 2,
        tickWidth: 2,
        tickLength: 8,
        tickColor: axisColor,
        min: kMin,
        max: kMax,
        plotLines: xGridPlotLines,
    }, {
        type: 'logarithmic',
        opposite: true,
        linkedTo: 0,
        tickPositions: scaleTickPositions,
        labels: {
            useHTML: true,
            formatter: function() {
                const position = Math.log10(this.value);
                const index = findClosestIndex(scaleTickPositions, position, 1e-6);
                return index === -1 ? '' : scaleTickLabels[index];
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            overflow: 'allow',
        },
        title: {
            text: latexInline('\\mathrm{Spatial\\ scale}\\;[\\mathrm{Mpc}/h]'),
            useHTML: true,
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        lineColor: axisColor,
        lineWidth: 2,
        tickWidth: 2,
        tickLength: 8,
        tickColor: axisColor,
        min: kMin,
        max: kMax,
    }];
    axisConfig.yAxis = {
        type: 'logarithmic',
        min: MATTER_Y_MIN,
        max: MATTER_Y_AXIS_MAX,
        tickPositions: yLogTicks,
        labels: {
            useHTML: true,
            formatter: function() {
                return formatLatexLogTickLabel(this.value);
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        title: {
            text: getMatterYAxisTitle(selectedOption.value),
            useHTML: true,
            rotation: 270,
            align: 'middle',
            margin: 0,
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        gridLineWidth: 0,
        plotLines: yGridPlotLines,
        lineWidth: 2,
        lineColor: axisColor,
        tickWidth: 2,
        tickLength: 8,
        tickColor: axisColor,
        alignTicks: false,
        endOnTick: false,
        startOnTick: false,
    };
    axisConfig.legend = {
        enabled: Boolean(data && data.datasets && data.datasets.length),
        floating: true,
        align: 'left',
        verticalAlign: 'bottom',
        layout: 'vertical',
        x: marginLeft + 12,
        y: -marginBottom - 6,
        borderWidth: 1,
        borderColor: axisColor,
        borderRadius: 0,
        padding: compactChart ? 5 : 8,
        itemStyle: {
            color: axisColor,
            fontSize: compactChart ? '9px' : '12px',
        },
        itemHoverStyle: {
            color: axisColor,
        },
        backgroundColor: isDarkMode ? 'rgba(0, 0, 0, 0.82)' : 'rgba(255, 255, 255, 0.88)',
        symbolHeight: compactChart ? 7 : 8,
        symbolWidth: compactChart ? 10 : 12,
        itemMarginTop: 1,
        itemMarginBottom: 1,
    };
    resetMatterSeries(lineWidth);
}



function updateAxes() {
    if (isMatterMode()) {
        updateMatterAxes();
        return;
    }

    var xLabels = [];
    var outputTicks = {};
    var fontSize = '20px';
    var marginLeft = 80;
    var marginRight = 30;
    var marginBottom = 125;
    var marginTop = 100;
    var lineWidth = 6;
    var stemWidth = 3;
    var pointSize = 4;
    var yMin = 0;
    var yMax = 8400;
    var yTicks = [0, 2000, 4000, 6000, 8000];
    var yLabels = ['0', '2000', '4000', '6000', '8000'];
    var yLabelsReduced = ['0', '4000', '8000'];
    if (selectedOption.value === "TE") {
	yMin = -230;
	yMax = 230;
	yTicks = [-200, -100, 0, 100, 200];
	yLabels = ['-200', '-100', '0', '100', '200'];
	yLabelsReduced = ['-200', '0', '200'];
    } else if (selectedOption.value === "EE") {
	yMin = -2;
	yMax = 60;
	yTicks = [0, 20, 40, 60];
	yLabels = ['0', '20', '40', '60'];
	yLabelsReduced = ['0', '60'];
    } else if (selectedOption.value === "PP") {
	yMin = 0;
	yMax = 2.5;
	yTicks = [0, 0.5, 1, 1.5, 2, 2.5];
	yLabels = ['0', '0.5', '1', '1.5', '2', '2.5'];
	yLabelsReduced = ['0', '1', '2'];
    }
    let angleDegrees;
    if (document.getElementById('outputChart').offsetWidth > 500) {
	xLabels = [latexInline('10^{1}'), latexInline('10^{2}'), latexInline("500"), latexInline("1000"), latexInline("1500"), latexInline("2000"), latexInline("2500")];
	outputTicks = generateXTicks(fewTicks = false);
	angleDegrees = [90, 18, 1, 0.2, 0.1, 0.07];
    } else {
	xLabels = [latexInline('10^{1}'), latexInline('10^{2}'), latexInline("1000"), latexInline("2000")];
	outputTicks = generateXTicks(fewTicks = true);
	fontSize = '14px';
	marginLeft = 65;
	marginRight = 20;
	marginBottom = 115;
	marginTop = 90;
	lineWidth = 3;
	stemWidth = 2;
	pointSize = 2.5;
	angleDegrees = [90, 18, 1, 0.1];
    }
    const { tickPositions: angleLabeledPositions, tickLabels: angleLabeledLabels } = generateAngleTicks(angleDegrees);
    const xMajorTicks = outputTicks.majorTicks;
    const xMinorTicks = outputTicks.minorTicks;
    const xMinorLabels = [];
    for (let i = 0; i < xMinorTicks.length; i++) {
        xMinorLabels.push("");
    }
    const { sortedValues: xTotalTicks, sortedLabels: xTotalLabels } = mergeSortAndSyncArrays(xMajorTicks, xLabels, xMinorTicks, xMinorLabels);

    // Reuse every bottom-axis tick position (major+minor) as an unlabeled backdrop of
    // degree ticks, then overlay the "nice" labeled degree ticks at their own positions.
    const angleUnlabeledLabels = xTotalTicks.map(() => "");
    const { sortedValues: angleTickPositions, sortedLabels: angleTickLabels } = mergeSortAndSyncArrays(
        angleLabeledPositions, angleLabeledLabels, xTotalTicks, angleUnlabeledLabels
    );

    if (document.getElementById('outputChart').offsetHeight < 400) {
	yLabels = yLabelsReduced;
	fontSize = '14px';
        marginLeft = 65;
        marginRight = 20;
        marginBottom = 115;
        marginTop = 90;
	lineWidth = 3;
	stemWidth = 2;
	pointSize = 2.5;
    }

    
    const isDarkMode = document.body.classList.contains('dark-mode');
    const axisColor = isDarkMode ? 'white' : 'black';
    const gridColor = getGridLineColor(isDarkMode);
    const xGridPlotLines = buildGridPlotLines(xMajorTicks, gridColor);
    const yGridPlotLines = buildGridPlotLines(yTicks.filter(value => yLabels.includes(value.toString())), gridColor);

    axisConfig.chart = {
        type: 'spline',
        backgroundColor: isDarkMode ? '#000000' : '#FFFFFF',
        animation: false, // Disable animation
        marginLeft: marginLeft,
        marginRight: marginRight,
        marginBottom: marginBottom,
        marginTop: marginTop,
        plotBorderWidth: 2,
        plotBorderColor: axisColor,
    };
    axisConfig.title = {
	text: ''
    };
    axisConfig.xAxis = [{
        type: 'linear',
        tickPositions: xTotalTicks,
        labels: {
            useHTML: true,
            formatter: function() {
                return xTotalLabels[xTotalTicks.indexOf(this.value)] || '';
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            rotation: 0,
            overflow: 'allow',
        },
        title: {
            text: latexInline('\\mathrm{Multipole},\\ \\ell'),
            useHTML: true,
            align: 'middle',
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        lineColor: axisColor,
        lineWidth: 2,
        tickWidth: 2,
        tickLength: 8,
        tickColor: axisColor,
        min: 0,
        max: 1,
        plotLines: xGridPlotLines,
    }, {
        type: 'linear',
        opposite: true,
        linkedTo: 0,
        tickPositions: angleTickPositions,
        labels: {
            useHTML: true,
            formatter: function() {
                const index = angleTickPositions.indexOf(this.value);
                return index === -1 ? '' : angleTickLabels[index];
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            rotation: 0,
            overflow: 'allow',
        },
        title: {
            text: latexInline('\\mathrm{Angular\\ scale}'),
            useHTML: true,
            align: 'middle',
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        lineColor: axisColor,
        lineWidth: 2,
        tickWidth: 2,
        tickLength: 8,
        tickColor: axisColor,
        min: 0,
        max: 1,
    }];
    axisConfig.yAxis = {
        tickPositions: yTicks,
        labels: {
            useHTML: true,
            formatter: function() {
                return yLabels.includes(this.value.toString()) ? latexInline(formatLatexNumber(this.value)) : '';
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            rotation: 270,
        },
        title: {
            text: getCmbYAxisTitle(selectedOption.value),
            useHTML: true,
            rotation: 270,
            align: 'middle',
            margin: 0,
            style: {
                color: axisColor,
                fontSize: fontSize
            },
        },
        gridLineWidth: 0,
        plotLines: yGridPlotLines,
        lineWidth: 2,
        lineColor: axisColor,
        tickWidth: 2,
        tickLength: 10,
        tickColor: axisColor,
        alignTicks: false,
        endOnTick: false,
        startOnTick: false,
        min: yMin,
        max: yMax,
    };
    axisConfig.legend = {
        enabled: false
    };
    resetCmbSeries(lineWidth, pointSize, stemWidth);
};
    
function getSliderNumber(sliderId) {
    return parseFloat(sliderById[sliderId].value);
}

function getCmbInput() {
    return [
        getSliderNumber("slider1"),
        getSliderNumber("slider2"),
        getSliderNumber("slider3"),
        getSliderNumber("slider4"),
        getSliderNumber("slider5"),
        getSliderNumber("slider6"),
    ];
}

function getMatterLinearInput() {
    return [
        getSliderNumber("slider1"),
        getSliderNumber("slider2"),
        getSliderNumber("slider3") / 100,
        getSliderNumber("slider5"),
        getSliderNumber("slider4"),
        getSliderNumber("slider7"),
    ];
}

function getMatterNonlinearInput() {
    return [
        getSliderNumber("slider1"),
        getSliderNumber("slider2"),
        getSliderNumber("slider3") / 100,
        getSliderNumber("slider5"),
        getSliderNumber("slider4"),
        getSliderNumber("slider8"),
        getSliderNumber("slider9"),
        getSliderNumber("slider7"),
    ];
}

function getMatterHubbleParameter() {
    return getSliderNumber("slider3") / 100;
}

function getDisplayedMatterKGrid() {
    const h = getMatterHubbleParameter();
    return matterKGrid.map(k => k / h);
}

async function predictModel(graphModel, inputValues) {
    const inputData = tf.tensor([inputValues]);
    const outputData = graphModel.predict(inputData);
    const outputArray = await outputData.array();
    inputData.dispose();
    outputData.dispose();
    return outputArray[0];
}

function updateAllSliderReadouts() {
    for (const slider of sliders) {
        updateSliderReadout(slider);
    }
}

async function updateCmbDataAndChart() {
    const outputArray = await predictModel(models.cmb, getCmbInput());
    var slice0 = 0;
    var slice1 = 100;
    var factor = (10 ** 6 * 2.7255) ** 2;

    if (selectedOption.value === "TT") {
	slice0 = 0;
	slice1 = 100;
    } else if (selectedOption.value === "TE") {
	slice0 = 100;
	slice1 = 200;
    } else if (selectedOption.value === "EE") {
	slice0 = 200;
	slice1 = 300;
    } else if (selectedOption.value === "PP") {
	slice0 = 300;
	slice1 = 400;
	factor = 1e+7;
    }
    const spectrum = outputArray.slice(slice0, slice1).map(x => x * factor);

    const xdata = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,23,25,27,30,33,36,40,44,49,54,60,67,75,83,92,103,115,128,143,160,179,200,223,249,278,311,348,387,426,465,504,543,582,621,660,699,738,777,816,855,894,933,972,1011,1050,1089,1128,1167,1206,1245,1284,1323,1362,1401,1440,1479,1518,1557,1596,1635,1674,1713,1752,1791,1830,1869,1908,1947,1986,2025,2064,2103,2142,2181,2220,2259,2298,2337,2376,2415,2454,2493]

    // Array af log-equidistant points from 2 to 2500
    const numPoints = 600;
    const startLog = Math.log10(2);
    const endLog = Math.log10(2500);
    const stepSize = (endLog - startLog) / (numPoints - 1);
    const xGrid = Array.from({ length: numPoints }, (_, index) => Math.pow(10, startLog + index * stepSize));
    // Interpolate the data
    const interpolatedData = interpolateData(spectrum, xdata, xGrid);

    if (data) {
	var seriesData = [];
	var errorData = [];
	if (selectedOption.value === "PP") {
	    seriesData = data.map(d => [scaleGraphPoint(d[0]), d[1] * 1e+7]);
	    errorData = data.map(d => [scaleGraphPoint(d[0]), d[1] * 1e+7 - d[2] * 1e+7, d[1] * 1e+7 + d[3] * 1e+7]);
	} else {
	    seriesData = data.map(d => [scaleGraphPoint(d[0]), d[1]]);
	    errorData = data.map(d => [scaleGraphPoint(d[0]), d[1] - d[2], d[1] + d[3]]);
	};
	axisConfig.series[1].data = seriesData;
	axisConfig.series[2].data = errorData;
    } else {
	axisConfig.series[1].data = [];
	axisConfig.series[2].data = [];
    };

    axisConfig.series[0].data = interpolatedData.map((y, i) => [scaleGraphPoint(xGrid[i]), y]);
}

function getMatterCompilationDatasets() {
    if (!data || !data.datasets || !isMatterMode()) {
        return [];
    }

    return data.datasets;
}

function clampPositive(value, fallback) {
    return Number.isFinite(value) && value > 0 ? value : fallback;
}

function createMatterCompilationSeries() {
    const series = [];

    for (const dataset of getMatterCompilationDatasets()) {
        const color = dataset.color || '#666666';
        const points = (dataset.points || [])
            .filter(point => point.k > 0 && point.pk > 0)
            .map(point => [point.k, point.pk]);
        const yErrors = (dataset.points || [])
            .filter(point => point.k > 0 && point.pk > 0)
            .map(point => [
                point.k,
                clampPositive(point.pkLow, point.pk),
                clampPositive(point.pkHigh, point.pk),
            ]);

        series.push({
            type: 'scatter',
            name: dataset.label,
            data: points,
            color,
            opacity: 0.85,
            marker: {
                symbol: 'circle',
                radius: 3.2,
                fillColor: color,
                lineColor: color,
                lineWidth: 1,
            },
            showInLegend: true,
        });

        series.push({
            type: 'errorbar',
            data: yErrors,
            linkedTo: ':previous',
            showInLegend: false,
            color,
            stemColor: color,
            whiskerColor: color,
            stemWidth: 1.5,
            whiskerLength: 6,
            opacity: 0.8,
            pointRange: 0,
            pointWidth: 3,
        });
    }

    return series;
}

function drawMatterXErrorBars(chart) {
    if (!isMatterMode() || !data || !data.datasets || !chart || !chart.renderer) {
        return;
    }

    const xAxis = chart.xAxis && chart.xAxis[0];
    const yAxis = chart.yAxis && chart.yAxis[0];

    if (!xAxis || !yAxis) {
        return;
    }

    const group = chart.renderer.g('matter-x-errorbars')
        .attr({ zIndex: 4 })
        .add();
    const xMin = xAxis.options.min;
    const xMax = xAxis.options.max;
    const yMin = yAxis.options.min;
    const yMax = yAxis.options.max;
    const capHalfHeight = 4;

    for (const dataset of data.datasets) {
        const color = dataset.color || '#666666';

        for (const point of dataset.points || []) {
            if (!Number.isFinite(point.kLow) || !Number.isFinite(point.kHigh) ||
                !Number.isFinite(point.pk) || point.kHigh <= 0 || point.kLow <= 0 ||
                point.pk <= 0 || point.pk < yMin || point.pk > yMax) {
                continue;
            }

            const low = Math.max(point.kLow, xMin);
            const high = Math.min(point.kHigh, xMax);

            if (low > high) {
                continue;
            }

            const xLow = xAxis.toPixels(low);
            const xHigh = xAxis.toPixels(high);
            const y = yAxis.toPixels(point.pk);

            chart.renderer.path([
                'M', xLow, y,
                'L', xHigh, y,
                'M', xLow, y - capHalfHeight,
                'L', xLow, y + capHalfHeight,
                'M', xHigh, y - capHalfHeight,
                'L', xHigh, y + capHalfHeight,
            ])
                .attr({
                    stroke: color,
                    'stroke-width': 1.5,
                    opacity: 0.8,
                })
                .add(group);
        }
    }
}

async function updateMatterDataAndChart() {
    const outputArray = selectedOption.value === "PK_LIN"
        ? await predictModel(models.matterLinear, getMatterLinearInput())
        : await predictModel(models.matterNonlinear, getMatterNonlinearInput());
    const displayedMatterKGrid = getDisplayedMatterKGrid();
    const volumeUnitScale = Math.pow(getMatterHubbleParameter(), 3);

    axisConfig.series[0].data = outputArray
        .map((y, i) => [displayedMatterKGrid[i], y * volumeUnitScale])
        .filter(point => point[0] > 0 && point[1] > 0);
    axisConfig.series = [axisConfig.series[0], ...createMatterCompilationSeries()];
}

async function updateDataAndChart() {
    updateAllSliderReadouts();

    if (isMatterMode()) {
        await updateMatterDataAndChart();
    } else {
        await updateCmbDataAndChart();
    }

    displayChart();
};

function displayChart() {
    Highcharts.setOptions({
	plotOptions: {
	    series: {
		animation: false,
		states: {
                    hover: {
			enabled: false
                    },
		    inactive: {
			enabled: false
		    }
		},
	    },
	}
    });
    const chart = Highcharts.chart('outputChart', axisConfig);
    drawMatterXErrorBars(chart);
    typesetMathLabels();
}


// Cubic spline interpolation function
function interpolateData(yData, xData, xGrid) {
    const n = yData.length;
    if (n < 2) {
        throw new Error('Data array must have at least 2 points for interpolation.');
    }

    // Prepare arrays for the spline interpolation
    const x = [...xData]; // Copy xData array
    const y = [...yData]; // Copy yData array

    // Calculate coefficients for the cubic spline
    const a = [];
    const b = [];
    const c = [];
    const d = [];
    const h = [];
    const alpha = [];
    const l = [];
    const mu = [];
    const z = [];

    // Step 1: Compute h[i] = x[i+1] - x[i]
    for (let i = 0; i < n - 1; i++) {
        h[i] = x[i + 1] - x[i];
    }

    // Step 2: Compute alpha[i] = (3/h[i]) * (y[i+1] - y[i]) - (3/h[i-1]) * (y[i] - y[i-1])
    for (let i = 1; i < n - 1; i++) {
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    // Step 3: Compute l and mu
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    for (let i = 1; i < n - 1; i++) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    // Step 4: Compute l[n], z[n], and c
    l[n - 1] = 1;
    z[n - 1] = 0;
    c[n - 1] = 0;

    for (let j = n - 2; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        a[j] = y[j];
    }

    // Step 5: Evaluate the spline at each desired point
    const interpolatedData = [];
    for (let i = 0; i < xGrid.length; i++) {
        const xValue = xGrid[i];
        let j = 0;
        // Find the segment to interpolate
        while (x[j] < xValue && j < n - 1) {
            j++;
        }
        if (j > 0) j--; // Ensure j is within bounds

        const dt = xValue - x[j];
        const interpolatedValue = a[j] + b[j] * dt + c[j] * dt ** 2 + d[j] * dt ** 3;
        interpolatedData.push(interpolatedValue);
    }

    return interpolatedData;
}
