// Professional Hydrogen Infrastructure Analysis Platform
class HydrogenAnalysisPlatform {
    constructor() {
        this.map = null;
        this.drawControl = null;
        this.drawnItems = null;
        this.satelliteLayer = null;
        this.currentTileLayer = null;
        this.isMapInitialized = false;
        this.currentAnalysis = null;
        this.analysisChart = null;
        this.activeDrawTool = null;
        
        // Analysis weights
        this.weights = {
            landCost: 25,
            renewablePotential: 30,
            infrastructure: 20,
            waterAvailability: 15,
            regulatory: 10
        };
        
        // Analysis data from the provided JSON
        this.analysisData = {
            "suitability_criteria": {
                "land_cost": {
                    "rajasthan": {"min": 150000, "avg": 250000, "max": 350000, "unit": "INR/hectare"},
                    "gujarat": {"min": 300000, "avg": 450000, "max": 600000, "unit": "INR/hectare"},
                    "maharashtra": {"min": 500000, "avg": 750000, "max": 1000000, "unit": "INR/hectare"},
                    "tamilnadu": {"min": 400000, "avg": 650000, "max": 900000, "unit": "INR/hectare"},
                    "karnataka": {"min": 350000, "avg": 550000, "max": 750000, "unit": "INR/hectare"},
                    "andhrapradesh": {"min": 280000, "avg": 420000, "max": 560000, "unit": "INR/hectare"}
                },
                "renewable_potential": {
                    "solar_irradiance": {
                        "rajasthan": {"value": 5.5, "rating": "Excellent", "capacity_factor": 0.22},
                        "gujarat": {"value": 5.2, "rating": "Very Good", "capacity_factor": 0.20},
                        "maharashtra": {"value": 4.8, "rating": "Good", "capacity_factor": 0.18},
                        "tamilnadu": {"value": 4.9, "rating": "Good", "capacity_factor": 0.19},
                        "karnataka": {"value": 5.1, "rating": "Very Good", "capacity_factor": 0.19},
                        "andhrapradesh": {"value": 5.0, "rating": "Good", "capacity_factor": 0.19}
                    },
                    "wind_potential": {
                        "tamilnadu": {"value": 7.5, "rating": "Excellent", "capacity_factor": 0.35},
                        "gujarat": {"value": 7.2, "rating": "Excellent", "capacity_factor": 0.33},
                        "rajasthan": {"value": 6.8, "rating": "Very Good", "capacity_factor": 0.30},
                        "karnataka": {"value": 6.5, "rating": "Very Good", "capacity_factor": 0.28},
                        "maharashtra": {"value": 6.2, "rating": "Good", "capacity_factor": 0.26},
                        "andhrapradesh": {"value": 6.0, "rating": "Good", "capacity_factor": 0.25}
                    }
                }
            },
            "benchmark_sites": [
                {
                    "name": "Gujarat Solar Park Reference",
                    "location": "Gujarat",
                    "score": 85,
                    "land_cost": 450000,
                    "renewable_score": 92,
                    "infrastructure_score": 88,
                    "water_score": 75,
                    "regulatory_score": 82,
                    "investment_estimate": 50000000,
                    "capacity_mw": 100,
                    "roi_years": 8.5
                }
            ]
        };
        
        this.init();
    }

    async init() {
        try {
            // Wait for DOM to be fully loaded
            if (document.readyState === 'loading') {
                await new Promise(resolve => {
                    document.addEventListener('DOMContentLoaded', resolve);
                });
            }
            
            this.setupEventListeners();
            await this.initializeMap();
            this.setupWeightControls();
            this.showToast('Platform Ready - Draw an area to begin analysis', 'success');
        } catch (error) {
            console.error('Platform initialization failed:', error);
            this.showToast('Platform initialization failed', 'error');
        }
    }

    setupEventListeners() {
        // Drawing tools
        this.addEventListenerSafe('draw-polygon', 'click', () => this.activateDrawTool('polygon'));
        this.addEventListenerSafe('draw-rectangle', 'click', () => this.activateDrawTool('rectangle'));
        this.addEventListenerSafe('draw-circle', 'click', () => this.activateDrawTool('circle'));
        this.addEventListenerSafe('clear-all', 'click', () => this.clearAllAreas());

        // Map controls
        this.addEventListenerSafe('reset-view', 'click', () => this.resetMapView());
        this.addEventListenerSafe('satellite-toggle', 'click', () => this.toggleSatelliteView());

        // Panel controls
        this.addEventListenerSafe('toggle-panel', 'click', () => this.togglePanel());
        this.addEventListenerSafe('close-results', 'click', () => this.hideResults());

        // Analysis preset
        this.addEventListenerSafe('analysis-preset', 'change', (e) => this.applyPreset(e.target.value));
        
        // Reset weights
        this.addEventListenerSafe('reset-weights', 'click', () => this.resetWeights());

        // Export functionality
        this.addEventListenerSafe('export-report', 'click', () => this.exportReport());
        
        // Help button
        this.addEventListenerSafe('help-btn', 'click', () => this.showHelp());
    }

    addEventListenerSafe(id, event, handler) {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener(event, handler);
        } else {
            console.warn(`Element with id '${id}' not found`);
        }
    }

    async initializeMap() {
        if (this.isMapInitialized) return;

        try {
            console.log('Initializing map...');
            
            // Wait for Leaflet to load
            if (typeof L === 'undefined') {
                await this.waitForLeaflet();
            }

            // Initialize map with retry logic
            let retries = 3;
            while (retries > 0 && !this.map) {
                try {
                    this.map = L.map('map', {
                        center: [20.5937, 78.9629], // Center of India
                        zoom: 5,
                        zoomControl: false,
                        preferCanvas: true
                    });
                    break;
                } catch (mapError) {
                    console.warn('Map creation attempt failed:', mapError);
                    retries--;
                    if (retries > 0) {
                        await this.delay(1000);
                    }
                }
            }

            if (!this.map) {
                throw new Error('Failed to create map after multiple attempts');
            }

            // Add zoom control to top-right
            L.control.zoom({ position: 'topright' }).addTo(this.map);

            // Add base tile layer with error handling
            this.currentTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors',
                maxZoom: 18,
                errorTileUrl: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgZmlsbD0iI2Y1ZjVmNSIvPjx0ZXh0IHg9IjEyOCIgeT0iMTI4IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIwLjNlbSI+TWFwIFRpbGUgTm90IEF2YWlsYWJsZTwvdGV4dD48L3N2Zz4='
            });

            this.currentTileLayer.addTo(this.map);

            // Initialize satellite layer
            this.satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: '© Esri',
                errorTileUrl: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgZmlsbD0iIzMzMzMzMyIvPjx0ZXh0IHg9IjEyOCIgeT0iMTI4IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiNjY2MiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIwLjNlbSI+U2F0ZWxsaXRlIFZpZXcgVW5hdmFpbGFibGU8L3RleHQ+PC9zdmc+'
            });

            // Initialize drawing
            this.drawnItems = new L.FeatureGroup();
            this.map.addLayer(this.drawnItems);
            
            await this.setupDrawControl();
            this.setupMapEvents();
            
            this.isMapInitialized = true;
            console.log('Map initialized successfully');
            
        } catch (error) {
            console.error('Map initialization failed:', error);
            this.showToast('Map initialization failed - please refresh the page', 'error');
            throw error;
        }
    }

    async waitForLeaflet() {
        let attempts = 0;
        while (typeof L === 'undefined' && attempts < 50) {
            await this.delay(100);
            attempts++;
        }
        if (typeof L === 'undefined') {
            throw new Error('Leaflet library failed to load');
        }
    }

    async setupDrawControl() {
        try {
            // Wait for Leaflet.draw to be available
            if (typeof L.Control.Draw === 'undefined') {
                await this.waitForLeafletDraw();
            }

            this.drawControl = new L.Control.Draw({
                position: 'topleft',
                draw: {
                    polygon: {
                        allowIntersection: false,
                        showArea: true,
                        drawError: {
                            color: '#e1e100',
                            message: '<strong>Cannot draw shape here</strong>'
                        },
                        shapeOptions: {
                            color: '#2563eb',
                            fillColor: '#2563eb',
                            fillOpacity: 0.2,
                            weight: 3
                        }
                    },
                    rectangle: {
                        showArea: true,
                        shapeOptions: {
                            color: '#10b981',
                            fillColor: '#10b981',
                            fillOpacity: 0.2,
                            weight: 3
                        }
                    },
                    circle: {
                        showRadius: true,
                        shapeOptions: {
                            color: '#f59e0b',
                            fillColor: '#f59e0b',
                            fillOpacity: 0.2,
                            weight: 3
                        }
                    },
                    marker: false,
                    circlemarker: false,
                    polyline: false
                },
                edit: {
                    featureGroup: this.drawnItems,
                    remove: true
                }
            });

            this.map.addControl(this.drawControl);
            
            // Hide the default toolbar immediately
            setTimeout(() => {
                const toolbars = document.querySelectorAll('.leaflet-draw-toolbar');
                toolbars.forEach(toolbar => {
                    toolbar.style.display = 'none';
                });
            }, 0);
            
        } catch (error) {
            console.error('Draw control setup failed:', error);
            this.showToast('Drawing tools setup failed', 'warning');
        }
    }

    async waitForLeafletDraw() {
        let attempts = 0;
        while (typeof L.Control.Draw === 'undefined' && attempts < 50) {
            await this.delay(100);
            attempts++;
        }
        if (typeof L.Control.Draw === 'undefined') {
            throw new Error('Leaflet.draw library failed to load');
        }
    }

    setupMapEvents() {
        if (!this.map) return;

        this.map.on('draw:created', async (event) => {
            try {
                const layer = event.layer;
                const type = event.layerType;
                
                // Add unique ID to layer
                layer._aoiId = 'aoi_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
                layer._aoiType = type;
                
                this.drawnItems.addLayer(layer);
                
                // Deactivate drawing tools
                this.deactivateAllDrawTools();
                
                this.showToast(`${type.charAt(0).toUpperCase() + type.slice(1)} area created - Analyzing...`, 'success');
                
                // Analyze the area
                await this.analyzeArea(layer);
                
            } catch (error) {
                console.error('Error handling draw:created event:', error);
                this.showToast('Error creating area', 'error');
            }
        });

        this.map.on('draw:edited', async () => {
            this.showToast('Area updated - reanalyzing...', 'info');
            // Re-analyze if there's a current analysis
            if (this.currentAnalysis) {
                await this.analyzeArea(this.currentAnalysis.layer);
            }
        });

        this.map.on('draw:deleted', () => {
            this.hideResults();
            this.currentAnalysis = null;
            this.showToast('Area deleted', 'info');
        });

        // Handle map load errors
        this.map.on('tileerror', (e) => {
            console.warn('Tile load error:', e);
        });
    }

    setupWeightControls() {
        const weights = ['land', 'renewable', 'infrastructure', 'water', 'regulatory'];
        
        weights.forEach(weight => {
            const slider = document.getElementById(`weight-${weight}`);
            if (slider) {
                slider.addEventListener('input', (e) => {
                    this.updateWeight(weight, parseInt(e.target.value));
                });
            }
        });
        
        this.updateTotalWeight();
    }

    updateWeight(type, value) {
        const mapping = {
            land: 'landCost',
            renewable: 'renewablePotential',
            infrastructure: 'infrastructure',
            water: 'waterAvailability',
            regulatory: 'regulatory'
        };
        
        this.weights[mapping[type]] = value;
        
        // Update display
        const valueElement = document.getElementById(`weight-${type}-value`);
        if (valueElement) {
            valueElement.textContent = `${value}%`;
        }
        
        this.updateTotalWeight();
        
        // Re-analyze if there's a current analysis
        if (this.currentAnalysis) {
            this.performAnalysisCalculations(this.currentAnalysis.layer)
                .then(analysis => {
                    this.currentAnalysis = analysis;
                    this.showResults(analysis);
                })
                .catch(error => {
                    console.error('Error re-analyzing:', error);
                });
        }
    }

    updateTotalWeight() {
        const total = Object.values(this.weights).reduce((sum, weight) => sum + weight, 0);
        const totalElement = document.getElementById('total-weight');
        if (totalElement) {
            totalElement.textContent = `${total}%`;
            totalElement.style.color = total === 100 ? 'var(--color-success)' : 'var(--color-warning)';
        }
    }

    resetWeights() {
        this.weights = {
            landCost: 25,
            renewablePotential: 30,
            infrastructure: 20,
            waterAvailability: 15,
            regulatory: 10
        };
        
        // Update sliders
        const updates = {
            'weight-land': 25,
            'weight-renewable': 30,
            'weight-infrastructure': 20,
            'weight-water': 15,
            'weight-regulatory': 10
        };
        
        Object.entries(updates).forEach(([id, value]) => {
            const slider = document.getElementById(id);
            const display = document.getElementById(`${id}-value`);
            
            if (slider) slider.value = value;
            if (display) display.textContent = `${value}%`;
        });
        
        this.updateTotalWeight();
        this.showToast('Weights reset to default', 'info');
    }

    activateDrawTool(tool) {
        if (!this.map || !this.drawControl) {
            this.showToast('Drawing tools not ready yet', 'warning');
            return;
        }

        try {
            // Reset all tool buttons
            document.querySelectorAll('.tool-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Activate selected tool button
            const toolBtn = document.getElementById(`draw-${tool}`);
            if (toolBtn) {
                toolBtn.classList.add('active');
            }
            
            // Deactivate any currently active tools first
            this.deactivateAllDrawTools();
            
            // Get the handler for the selected tool
            const handlers = this.drawControl._toolbars.draw._modes;
            const handler = handlers[tool]?.handler;
            
            if (handler) {
                handler.enable();
                this.activeDrawTool = tool;
                this.showToast(`${tool} drawing tool activated - Click on the map to start drawing`, 'info');
            } else {
                throw new Error(`Handler for ${tool} not found`);
            }
            
        } catch (error) {
            console.error('Error activating draw tool:', error);
            this.showToast('Error activating drawing tool', 'error');
        }
    }

    deactivateAllDrawTools() {
        try {
            if (!this.drawControl || !this.drawControl._toolbars) return;
            
            const handlers = this.drawControl._toolbars.draw._modes;
            Object.values(handlers).forEach(mode => {
                if (mode.handler && mode.handler.enabled && mode.handler.enabled()) {
                    mode.handler.disable();
                }
            });
            
            // Reset tool buttons
            document.querySelectorAll('.tool-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            this.activeDrawTool = null;
            
        } catch (error) {
            console.error('Error deactivating draw tools:', error);
        }
    }

    async analyzeArea(layer) {
        try {
            this.showLoading();
            
            // Simulate processing time
            await this.delay(1500);
            
            const analysis = await this.performAnalysisCalculations(layer);
            this.currentAnalysis = analysis;
            
            this.updateLayerStyle(layer, analysis);
            this.showResults(analysis);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showToast('Analysis failed', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async performAnalysisCalculations(layer) {
        const bounds = layer.getBounds();
        const center = bounds.getCenter();
        const area = this.calculateArea(layer);
        const state = this.getStateFromCoordinates(center.lat, center.lng);
        
        // Calculate individual scores
        const scores = {
            landCost: this.calculateLandCostScore(state, area),
            renewablePotential: this.calculateRenewableScore(state, center),
            infrastructure: this.calculateInfrastructureScore(center),
            waterAvailability: this.calculateWaterScore(center),
            regulatory: this.calculateRegulatoryScore(state)
        };
        
        // Calculate weighted overall score
        const overallScore = this.calculateWeightedScore(scores);
        const suitability = this.getSuitabilityRating(overallScore);
        
        return {
            layer: layer,
            center: center,
            area: area,
            state: state,
            scores: scores,
            overallScore: overallScore,
            suitability: suitability,
            investment: this.calculateInvestmentEstimate(area),
            roi: this.calculateROI(overallScore, area),
            landCostRange: this.getLandCostRange(state),
            timestamp: new Date()
        };
    }

    calculateLandCostScore(state, area) {
        const stateData = this.analysisData.suitability_criteria.land_cost[state.toLowerCase()];
        if (stateData) {
            // Lower cost = higher score
            const avgCost = stateData.avg;
            const maxCost = 1000000; // Maximum expected cost
            return Math.max(10, 100 - (avgCost / maxCost) * 100);
        }
        return 60; // Default score
    }

    calculateRenewableScore(state, center) {
        const stateKey = state.toLowerCase();
        const solarData = this.analysisData.suitability_criteria.renewable_potential.solar_irradiance[stateKey];
        const windData = this.analysisData.suitability_criteria.renewable_potential.wind_potential[stateKey];
        
        let score = 50; // Base score
        
        if (solarData) {
            score += (solarData.value / 6.0) * 30; // Solar contribution
        }
        
        if (windData) {
            score += (windData.value / 8.0) * 20; // Wind contribution
        }
        
        // Coastal bonus for wind
        if (this.isCoastalLocation(center)) {
            score += 10;
        }
        
        return Math.min(score, 100);
    }

    calculateInfrastructureScore(center) {
        let score = 70; // Base infrastructure score
        
        if (this.isNearMajorCity(center)) {
            score += 15;
        }
        
        if (this.isInIndustrialCorridor(center)) {
            score += 10;
        }
        
        return Math.min(score, 100);
    }

    calculateWaterScore(center) {
        let score = 65; // Base water score
        
        if (this.isCoastalLocation(center)) {
            score += 15;
        }
        
        if (this.isNearRiver(center)) {
            score += 10;
        }
        
        if (this.isDesertArea(center)) {
            score -= 20;
        }
        
        return Math.max(20, Math.min(score, 100));
    }

    calculateRegulatoryScore(state) {
        const stateScores = {
            gujarat: 90,
            rajasthan: 85,
            maharashtra: 80,
            tamilnadu: 82,
            karnataka: 78,
            andhrapradesh: 75
        };
        
        return stateScores[state.toLowerCase()] || 70;
    }

    calculateWeightedScore(scores) {
        let weightedSum = 0;
        let totalWeight = 0;
        
        Object.entries(scores).forEach(([criterion, score]) => {
            const weight = this.weights[criterion] || 0;
            weightedSum += score * (weight / 100);
            totalWeight += weight;
        });
        
        return totalWeight > 0 ? Math.round((weightedSum / totalWeight) * 100) : 0;
    }

    getSuitabilityRating(score) {
        if (score >= 80) return { rating: 'Excellent', class: 'excellent' };
        if (score >= 65) return { rating: 'Good', class: 'good' };
        if (score >= 45) return { rating: 'Moderate', class: 'moderate' };
        return { rating: 'Poor', class: 'poor' };
    }

    calculateInvestmentEstimate(area) {
        return Math.round(area * 500);
    }

    calculateROI(score, area) {
        const baseROI = 15;
        const scoreBonus = (score - 50) / 10;
        return Math.max(6, Math.round(baseROI - scoreBonus));
    }

    getLandCostRange(state) {
        const stateData = this.analysisData.suitability_criteria.land_cost[state.toLowerCase()];
        if (stateData) {
            return `${(stateData.min / 100000).toFixed(1)}-${(stateData.max / 100000).toFixed(1)} L/ha`;
        }
        return '3-8 L/ha';
    }

    calculateArea(layer) {
        if (layer instanceof L.Circle) {
            const radius = layer.getRadius(); // in meters
            return Math.PI * Math.pow(radius / 1000, 2); // Convert to km²
        } else if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
            // Simplified area calculation
            const bounds = layer.getBounds();
            const sw = bounds.getSouthWest();
            const ne = bounds.getNorthEast();
            const width = this.getDistanceFromLatLonInKm(sw.lat, sw.lng, sw.lat, ne.lng);
            const height = this.getDistanceFromLatLonInKm(sw.lat, sw.lng, ne.lat, sw.lng);
            return width * height * 0.7; // Approximate for polygon shape
        }
        return 10; // Default area
    }

    getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) {
        const R = 6371; // Radius of the earth in km
        const dLat = this.deg2rad(lat2 - lat1);
        const dLon = this.deg2rad(lon2 - lon1);
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(this.deg2rad(lat1)) * Math.cos(this.deg2rad(lat2)) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    deg2rad(deg) {
        return deg * (Math.PI / 180);
    }

    getStateFromCoordinates(lat, lng) {
        if (lng >= 68.5 && lng <= 74.5 && lat >= 20.0 && lat <= 26.0) return 'Gujarat';
        if (lng >= 69.0 && lng <= 78.0 && lat >= 24.0 && lat <= 30.5) return 'Rajasthan';
        if (lng >= 72.0 && lng <= 80.0 && lat >= 15.5 && lat <= 22.0) return 'Maharashtra';
        if (lng >= 76.0 && lng <= 84.0 && lat >= 8.0 && lat <= 15.0) return 'TamilNadu';
        if (lng >= 74.0 && lng <= 78.0 && lat >= 11.5 && lat <= 19.0) return 'Karnataka';
        if (lng >= 77.0 && lng <= 84.5 && lat >= 12.5 && lat <= 19.5) return 'AndhraPradesh';
        return 'Gujarat'; // Default state
    }

    isCoastalLocation(center) {
        return (center.lng < 75 && center.lat < 23) || 
               (center.lng > 79 && center.lat < 20) || 
               (center.lat < 12);
    }

    isNearMajorCity(center) {
        const cities = [
            { lat: 19.0760, lng: 72.8777 }, // Mumbai
            { lat: 28.7041, lng: 77.1025 }, // Delhi  
            { lat: 12.9716, lng: 77.5946 }, // Bangalore
            { lat: 13.0827, lng: 80.2707 }, // Chennai
            { lat: 22.5726, lng: 88.3639 }, // Kolkata
            { lat: 23.0225, lng: 72.5714 }  // Ahmedabad
        ];
        
        return cities.some(city => {
            const distance = this.getDistanceFromLatLonInKm(center.lat, center.lng, city.lat, city.lng);
            return distance < 100;
        });
    }

    isInIndustrialCorridor(center) {
        return (center.lat >= 19 && center.lat <= 29 && center.lng >= 72 && center.lng <= 77);
    }

    isNearRiver(center) {
        return (center.lng >= 75 && center.lng <= 83 && center.lat >= 22 && center.lat <= 27) || 
               (center.lng >= 73 && center.lng <= 81 && center.lat >= 16 && center.lat <= 23);
    }

    isDesertArea(center) {
        return (center.lng >= 69 && center.lng <= 75 && center.lat >= 24 && center.lat <= 30);
    }

    updateLayerStyle(layer, analysis) {
        const colors = {
            excellent: '#059669',
            good: '#10b981', 
            moderate: '#f59e0b',
            poor: '#ef4444'
        };
        
        const color = colors[analysis.suitability.class];
        layer.setStyle({
            color: color,
            fillColor: color,
            fillOpacity: 0.3,
            weight: 3
        });
        
        // Add popup
        layer.bindPopup(this.createPopupContent(analysis), {
            maxWidth: 280,
            className: 'suitability-popup'
        });
    }

    createPopupContent(analysis) {
        return `
            <div class="suitability-popup">
                <div class="popup-title">Analysis Results</div>
                <div class="popup-score">
                    <div class="popup-score-value">${analysis.overallScore}</div>
                    <div class="popup-score-label">${analysis.suitability.rating}</div>
                </div>
                <div class="popup-details">
                    <strong>Area:</strong> ${analysis.area.toFixed(1)} km²<br>
                    <strong>State:</strong> ${analysis.state}<br>
                    <strong>Investment:</strong> ₹${analysis.investment} Cr<br>
                    <strong>ROI Timeline:</strong> ${analysis.roi} years
                </div>
                <div class="popup-action">
                    <button class="popup-btn" onclick="window.hydrogenPlatform.showDetailedResults()">
                        View Details
                    </button>
                </div>
            </div>
        `;
    }

    showResults(analysis) {
        const resultsPanel = document.getElementById('results-panel');
        if (!resultsPanel) return;
        
        // Update overall score
        const overallScoreEl = document.getElementById('overall-score');
        const ratingElement = document.getElementById('score-rating');
        
        if (overallScoreEl) overallScoreEl.textContent = analysis.overallScore;
        if (ratingElement) {
            ratingElement.textContent = analysis.suitability.rating;
            ratingElement.className = `score-rating ${analysis.suitability.class}`;
        }
        
        // Update location details
        this.updateElement('area-size', `${analysis.area.toFixed(1)} km²`);
        this.updateElement('primary-state', analysis.state);
        this.updateElement('investment-estimate', `₹${analysis.investment} Cr`);
        this.updateElement('roi-timeline', `${analysis.roi} years`);
        this.updateElement('land-cost-range', analysis.landCostRange);
        this.updateElement('renewable-score', `${Math.round(analysis.scores.renewablePotential)}/100`);
        
        // Update chart
        this.updateAnalysisChart(analysis.scores);
        
        // Update recommendations
        this.updateRecommendations(analysis);
        
        // Show panel
        resultsPanel.classList.remove('hidden');
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateAnalysisChart(scores) {
        const canvas = document.querySelector('#criteria-chart canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart
        if (this.analysisChart) {
            this.analysisChart.destroy();
        }
        
        // Wait for Chart.js to be available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded yet');
            return;
        }
        
        this.analysisChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Land Cost', 'Renewable', 'Infrastructure', 'Water', 'Regulatory'],
                datasets: [{
                    label: 'Score',
                    data: [
                        Math.round(scores.landCost),
                        Math.round(scores.renewablePotential),
                        Math.round(scores.infrastructure),
                        Math.round(scores.waterAvailability),
                        Math.round(scores.regulatory)
                    ],
                    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'],
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: (value) => value + '%' }
                    },
                    x: {
                        ticks: { maxRotation: 0 }
                    }
                }
            }
        });
    }

    updateRecommendations(analysis) {
        const container = document.getElementById('recommendations-list');
        if (!container) return;
        
        const recommendations = this.generateRecommendations(analysis);
        
        container.innerHTML = recommendations.map(rec => `
            <div class="recommendation-item ${rec.priority}">
                <div class="recommendation-title">${rec.title}</div>
                <div class="recommendation-desc">${rec.description}</div>
            </div>
        `).join('');
    }

    generateRecommendations(analysis) {
        const recommendations = [];
        const score = analysis.overallScore;
        
        if (score >= 80) {
            recommendations.push({
                title: 'Excellent Development Potential',
                description: 'This location shows exceptional suitability for hydrogen infrastructure. Consider fast-track development with priority funding.',
                priority: 'high'
            });
        } else if (score >= 65) {
            recommendations.push({
                title: 'Good Development Candidate',
                description: 'Strong potential for hydrogen infrastructure development. Recommended for detailed feasibility study.',
                priority: 'medium'
            });
        } else {
            recommendations.push({
                title: 'Consider Alternative Locations',
                description: 'Current suitability is limited. Evaluate nearby areas or wait for infrastructure improvements.',
                priority: 'low'
            });
        }
        
        const scores = analysis.scores;
        if (scores.renewablePotential < 60) {
            recommendations.push({
                title: 'Enhance Renewable Integration',
                description: 'Consider hybrid solar-wind systems or grid electricity with renewable certificates to improve viability.',
                priority: 'medium'
            });
        }
        
        if (scores.infrastructure < 60) {
            recommendations.push({
                title: 'Infrastructure Development Required',
                description: 'Coordinate with local authorities for grid connectivity and transportation infrastructure improvements.',
                priority: 'high'
            });
        }
        
        return recommendations;
    }

    showDetailedResults() {
        if (this.currentAnalysis) {
            this.showResults(this.currentAnalysis);
        }
    }

    hideResults() {
        const resultsPanel = document.getElementById('results-panel');
        if (resultsPanel) {
            resultsPanel.classList.add('hidden');
        }
    }

    togglePanel() {
        const panel = document.getElementById('control-panel');
        const toggleBtn = document.getElementById('toggle-panel');
        
        if (panel && toggleBtn) {
            panel.classList.toggle('collapsed');
            const icon = toggleBtn.querySelector('.toggle-icon');
            if (icon) {
                icon.textContent = panel.classList.contains('collapsed') ? '›' : '‹';
            }
        }
    }

    clearAllAreas() {
        if (this.drawnItems) {
            this.drawnItems.clearLayers();
            this.hideResults();
            this.currentAnalysis = null;
            this.deactivateAllDrawTools();
            this.showToast('All areas cleared', 'info');
        }
    }

    resetMapView() {
        if (this.map) {
            this.map.setView([20.5937, 78.9629], 5);
            this.showToast('Map view reset', 'info');
        }
    }

    toggleSatelliteView() {
        if (!this.satelliteLayer || !this.map) return;
        
        if (this.map.hasLayer(this.satelliteLayer)) {
            this.map.removeLayer(this.satelliteLayer);
            this.map.addLayer(this.currentTileLayer);
            this.showToast('Switched to map view', 'info');
        } else {
            this.map.removeLayer(this.currentTileLayer);
            this.map.addLayer(this.satelliteLayer);
            this.showToast('Switched to satellite view', 'info');
        }
    }

    applyPreset(preset) {
        const presets = {
            'cost-optimized': { landCost: 40, renewablePotential: 25, infrastructure: 15, waterAvailability: 10, regulatory: 10 },
            'renewable-focused': { landCost: 15, renewablePotential: 45, infrastructure: 20, waterAvailability: 10, regulatory: 10 },
            'infrastructure-focused': { landCost: 20, renewablePotential: 20, infrastructure: 35, waterAvailability: 15, regulatory: 10 }
        };
        
        if (presets[preset]) {
            this.weights = presets[preset];
            
            Object.entries(this.weights).forEach(([key, value]) => {
                const mapping = {
                    landCost: 'land',
                    renewablePotential: 'renewable',
                    infrastructure: 'infrastructure',
                    waterAvailability: 'water',
                    regulatory: 'regulatory'
                };
                
                const sliderKey = mapping[key];
                const slider = document.getElementById(`weight-${sliderKey}`);
                const display = document.getElementById(`weight-${sliderKey}-value`);
                
                if (slider) slider.value = value;
                if (display) display.textContent = `${value}%`;
            });
            
            this.updateTotalWeight();
            this.showToast(`Applied ${preset.replace('-', ' ')} preset`, 'success');
        }
    }

    exportReport() {
        if (!this.currentAnalysis) {
            this.showToast('No analysis data to export', 'warning');
            return;
        }
        
        const reportData = {
            timestamp: new Date().toISOString(),
            analysis: {
                location: {
                    center: this.currentAnalysis.center,
                    area: this.currentAnalysis.area,
                    state: this.currentAnalysis.state
                },
                scores: this.currentAnalysis.scores,
                overallScore: this.currentAnalysis.overallScore,
                suitability: this.currentAnalysis.suitability,
                investment: this.currentAnalysis.investment,
                roi: this.currentAnalysis.roi
            },
            weights: this.weights
        };
        
        const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hydrogen-analysis-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showToast('Analysis report exported successfully', 'success');
    }

    showHelp() {
        this.showToast('Help: Draw areas on the map using the tools in the left panel to get detailed suitability analysis', 'info');
    }

    showLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('hidden');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (container.contains(toast)) {
                    container.removeChild(toast);
                }
            }, 300);
        }, 4000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the platform when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.hydrogenPlatform = new HydrogenAnalysisPlatform();
});