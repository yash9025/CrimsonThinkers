// Hydrogen Infrastructure Intelligence Platform - Professional Implementation

class HydrogenIntelligencePlatform {
    constructor() {
        this.map = null;
        this.drawControl = null;
        this.drawnItems = null;
        this.currentAOI = null;
        this.markers = {
            plants: L.layerGroup(),
            storage: L.layerGroup(),
            demand: L.layerGroup(),
            renewable: L.layerGroup()
        };
        this.analysisCharts = {};
        this.isAnalyzing = false;
        
        // Application data
        this.data = {
            "platform_metadata": {
                "name": "Hydrogen Intelligence Platform",
                "version": "3.0.0",
                "accuracy_target": 0.89,
                "coverage": "Global",
                "update_frequency": "Real-time"
            },
            "hydrogen_infrastructure": {
                "plants": [
                    {"id": 1, "name": "Green Valley H2 Hub", "lat": 37.7749, "lng": -122.4194, "capacity": "150 MW", "status": "operational", "type": "electrolysis"},
                    {"id": 2, "name": "Desert Sun Facility", "lat": 34.0522, "lng": -118.2437, "capacity": "200 MW", "status": "construction", "type": "electrolysis"},
                    {"id": 3, "name": "Nordic Wind H2", "lat": 59.9139, "lng": 10.7522, "capacity": "300 MW", "status": "operational", "type": "electrolysis"},
                    {"id": 4, "name": "Texas Gulf Complex", "lat": 29.7604, "lng": -95.3698, "capacity": "500 MW", "status": "planning", "type": "electrolysis"},
                    {"id": 5, "name": "Australian Solar H2", "lat": -33.8688, "lng": 151.2093, "capacity": "250 MW", "status": "operational", "type": "electrolysis"}
                ],
                "storage_facilities": [
                    {"id": 1, "name": "Bay Area Storage", "lat": 37.8044, "lng": -122.2711, "capacity": "1000 tonnes", "status": "operational", "type": "compressed"},
                    {"id": 2, "name": "Rotterdam Terminal", "lat": 51.9244, "lng": 4.4777, "capacity": "2500 tonnes", "status": "operational", "type": "liquefied"},
                    {"id": 3, "name": "Tokyo Port Hub", "lat": 35.6762, "lng": 139.6503, "capacity": "1500 tonnes", "status": "construction", "type": "compressed"}
                ],
                "demand_centers": [
                    {"id": 1, "name": "Silicon Valley Industrial", "lat": 37.3861, "lng": -122.0839, "demand": 45000, "type": "technology", "growth": 0.15},
                    {"id": 2, "name": "Hamburg Steel Complex", "lat": 53.5511, "lng": 9.9937, "demand": 78000, "type": "steel", "growth": 0.12},
                    {"id": 3, "name": "Tokyo Manufacturing", "lat": 35.6895, "lng": 139.6917, "demand": 62000, "type": "electronics", "growth": 0.18},
                    {"id": 4, "name": "Houston Petrochemical", "lat": 29.7633, "lng": -95.3633, "demand": 95000, "type": "chemicals", "growth": 0.14}
                ],
                "renewable_sources": [
                    {"id": 1, "name": "California Solar Farm", "lat": 35.2828, "lng": -120.6596, "capacity": 500, "type": "solar", "efficiency": 0.22},
                    {"id": 2, "name": "North Sea Wind", "lat": 54.5, "lng": 3.0, "capacity": 800, "type": "offshore_wind", "efficiency": 0.45},
                    {"id": 3, "name": "Australian Outback Solar", "lat": -26.0, "lng": 135.0, "capacity": 1200, "type": "solar", "efficiency": 0.24},
                    {"id": 4, "name": "Texas Wind Corridor", "lat": 32.0, "lng": -101.0, "capacity": 900, "type": "onshore_wind", "efficiency": 0.38}
                ]
            },
            "analysis_criteria": {
                "renewable_proximity": {"weight": 0.20, "description": "Access to renewable energy sources"},
                "demand_access": {"weight": 0.18, "description": "Proximity to hydrogen demand centers"},
                "infrastructure": {"weight": 0.16, "description": "Existing transportation and storage infrastructure"},
                "regulatory": {"weight": 0.14, "description": "Policy support and regulatory environment"},
                "economic": {"weight": 0.12, "description": "Economic factors and development costs"},
                "environmental": {"weight": 0.10, "description": "Environmental impact and sustainability"},
                "technical": {"weight": 0.10, "description": "Technical feasibility and grid connectivity"}
            }
        };

        this.init();
    }

    async init() {
        this.initializeMap();
        this.setupDrawingTools();
        this.loadInfrastructureData();
        this.setupEventListeners();
        this.setupThemeToggle();
        this.initializeUI();
        
        console.log('Hydrogen Intelligence Platform initialized');
    }

    initializeMap() {
        // Initialize map
        this.map = L.map('map', {
            center: [40.7128, -74.0060],
            zoom: 3,
            zoomControl: true
        });

        // Add tile layers
        const baseLayers = {
            'Satellite': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }),
            'Terrain': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            })
        };

        baseLayers['Satellite'].addTo(this.map);

        // Initialize feature group for drawn items
        this.drawnItems = new L.FeatureGroup();
        this.map.addLayer(this.drawnItems);
    }

    setupDrawingTools() {
        // Drawing options
        const drawOptions = {
            position: 'topright',
            draw: {
                polyline: false,
                polygon: {
                    allowIntersection: false,
                    drawError: {
                        color: '#e1e100',
                        message: '<strong>Error:</strong> Shape edges cannot cross!'
                    },
                    shapeOptions: {
                        color: '#32808D',
                        weight: 3,
                        opacity: 0.8,
                        fillOpacity: 0.2
                    }
                },
                circle: {
                    shapeOptions: {
                        color: '#32808D',
                        weight: 3,
                        opacity: 0.8,
                        fillOpacity: 0.2
                    }
                },
                rectangle: {
                    shapeOptions: {
                        color: '#32808D',
                        weight: 3,
                        opacity: 0.8,
                        fillOpacity: 0.2
                    }
                },
                marker: false,
                circlemarker: false
            },
            edit: {
                featureGroup: this.drawnItems,
                remove: true
            }
        };

        this.drawControl = new L.Control.Draw(drawOptions);

        // Map drawing events
        this.map.on('draw:created', (e) => {
            this.handleDrawCreated(e);
        });

        this.map.on('draw:deleted', (e) => {
            this.handleDrawDeleted(e);
        });
    }

    handleDrawCreated(e) {
        const layer = e.layer;
        
        // Clear existing AOI
        this.drawnItems.clearLayers();
        this.drawnItems.addLayer(layer);
        this.currentAOI = layer;

        // Calculate AOI metrics
        const metrics = this.calculateAOIMetrics(layer);
        this.displayAOIInfo(metrics);

        // Trigger analysis
        this.runAnalysis();
    }

    handleDrawDeleted(e) {
        this.currentAOI = null;
        this.hideAOIInfo();
        this.clearAnalysis();
    }

    calculateAOIMetrics(layer) {
        let area = 0;
        let perimeter = 0;
        let center = null;

        if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
            const latlngs = layer.getLatLngs()[0];
            area = L.GeometryUtil ? L.GeometryUtil.geodesicArea(latlngs) : 0;
            center = layer.getBounds().getCenter();
            
            // Calculate perimeter
            for (let i = 0; i < latlngs.length; i++) {
                const next = (i + 1) % latlngs.length;
                perimeter += latlngs[i].distanceTo(latlngs[next]);
            }
        } else if (layer instanceof L.Circle) {
            const radius = layer.getRadius();
            area = Math.PI * radius * radius;
            perimeter = 2 * Math.PI * radius;
            center = layer.getLatLng();
        }

        return {
            area: (area / 1000000).toFixed(2) + ' km²', // Convert to km²
            perimeter: (perimeter / 1000).toFixed(2) + ' km', // Convert to km
            center: center ? `${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}` : '--'
        };
    }

    displayAOIInfo(metrics) {
        const aoiInfo = document.getElementById('aoiInfo');
        document.getElementById('aoiArea').textContent = metrics.area;
        document.getElementById('aoiPerimeter').textContent = metrics.perimeter;
        document.getElementById('aoiCenter').textContent = metrics.center;
        aoiInfo.style.display = 'block';
    }

    hideAOIInfo() {
        document.getElementById('aoiInfo').style.display = 'none';
    }

    loadInfrastructureData() {
        // Load production plants
        this.data.hydrogen_infrastructure.plants.forEach(plant => {
            const marker = this.createPlantMarker(plant);
            this.markers.plants.addLayer(marker);
        });

        // Load storage facilities
        this.data.hydrogen_infrastructure.storage_facilities.forEach(storage => {
            const marker = this.createStorageMarker(storage);
            this.markers.storage.addLayer(marker);
        });

        // Load demand centers
        this.data.hydrogen_infrastructure.demand_centers.forEach(demand => {
            const marker = this.createDemandMarker(demand);
            this.markers.demand.addLayer(marker);
        });

        // Load renewable sources
        this.data.hydrogen_infrastructure.renewable_sources.forEach(renewable => {
            const marker = this.createRenewableMarker(renewable);
            this.markers.renewable.addLayer(marker);
        });

        // Add all layers to map
        Object.values(this.markers).forEach(layer => {
            layer.addTo(this.map);
        });
    }

    createPlantMarker(plant) {
        const statusColors = {
            'operational': '#32C55E',
            'construction': '#F59E0B',
            'planning': '#6B7280'
        };

        return L.circleMarker([plant.lat, plant.lng], {
            radius: 8,
            fillColor: statusColors[plant.status] || '#6B7280',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }).bindPopup(`
            <div class="marker-popup">
                <h4>${plant.name}</h4>
                <p><strong>Capacity:</strong> ${plant.capacity}</p>
                <p><strong>Status:</strong> ${plant.status}</p>
                <p><strong>Type:</strong> ${plant.type}</p>
            </div>
        `);
    }

    createStorageMarker(storage) {
        return L.circleMarker([storage.lat, storage.lng], {
            radius: 6,
            fillColor: '#8B5CF6',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }).bindPopup(`
            <div class="marker-popup">
                <h4>${storage.name}</h4>
                <p><strong>Capacity:</strong> ${storage.capacity}</p>
                <p><strong>Status:</strong> ${storage.status}</p>
                <p><strong>Type:</strong> ${storage.type}</p>
            </div>
        `);
    }

    createDemandMarker(demand) {
        return L.circleMarker([demand.lat, demand.lng], {
            radius: Math.max(4, demand.demand / 10000),
            fillColor: '#EF4444',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.6
        }).bindPopup(`
            <div class="marker-popup">
                <h4>${demand.name}</h4>
                <p><strong>Demand:</strong> ${demand.demand.toLocaleString()} tons/year</p>
                <p><strong>Type:</strong> ${demand.type}</p>
                <p><strong>Growth:</strong> ${(demand.growth * 100).toFixed(1)}%</p>
            </div>
        `);
    }

    createRenewableMarker(renewable) {
        const typeColors = {
            'solar': '#F59E0B',
            'offshore_wind': '#06B6D4',
            'onshore_wind': '#10B981'
        };

        return L.circleMarker([renewable.lat, renewable.lng], {
            radius: Math.max(4, renewable.capacity / 100),
            fillColor: typeColors[renewable.type] || '#6B7280',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.7
        }).bindPopup(`
            <div class="marker-popup">
                <h4>${renewable.name}</h4>
                <p><strong>Capacity:</strong> ${renewable.capacity} MW</p>
                <p><strong>Type:</strong> ${renewable.type.replace('_', ' ')}</p>
                <p><strong>Efficiency:</strong> ${(renewable.efficiency * 100).toFixed(1)}%</p>
            </div>
        `);
    }

    setupEventListeners() {
        // Drawing tool buttons
        document.getElementById('drawPolygon').addEventListener('click', () => {
            new L.Draw.Polygon(this.map, this.drawControl.options.draw.polygon).enable();
            this.setActiveDrawingTool('drawPolygon');
        });

        document.getElementById('drawRectangle').addEventListener('click', () => {
            new L.Draw.Rectangle(this.map, this.drawControl.options.draw.rectangle).enable();
            this.setActiveDrawingTool('drawRectangle');
        });

        document.getElementById('drawCircle').addEventListener('click', () => {
            new L.Draw.Circle(this.map, this.drawControl.options.draw.circle).enable();
            this.setActiveDrawingTool('drawCircle');
        });

        document.getElementById('clearAOI').addEventListener('click', () => {
            this.drawnItems.clearLayers();
            this.currentAOI = null;
            this.hideAOIInfo();
            this.clearAnalysis();
        });

        // Layer toggles
        document.getElementById('plantLayer').addEventListener('change', (e) => {
            this.toggleLayer('plants', e.target.checked);
        });

        document.getElementById('storageLayer').addEventListener('change', (e) => {
            this.toggleLayer('storage', e.target.checked);
        });

        document.getElementById('demandLayer').addEventListener('change', (e) => {
            this.toggleLayer('demand', e.target.checked);
        });

        document.getElementById('renewableLayer').addEventListener('change', (e) => {
            this.toggleLayer('renewable', e.target.checked);
        });

        // Parameter sliders
        const sliders = ['renewableWeight', 'demandWeight', 'infraWeight', 'economicWeight'];
        sliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            slider.addEventListener('input', (e) => {
                const valueSpan = e.target.parentNode.querySelector('.parameter-value');
                valueSpan.textContent = e.target.value + '%';
                
                if (this.currentAOI) {
                    this.runAnalysis();
                }
            });
        });

        // Action buttons
        document.getElementById('runAnalysis').addEventListener('click', () => {
            if (this.currentAOI) {
                this.runAnalysis();
            } else {
                this.showNotification('Please draw an area of interest first.', 'warning');
            }
        });

        document.getElementById('exportData').addEventListener('click', () => {
            this.exportAnalysisData();
        });

        // Close AOI info
        document.getElementById('closeAoiInfo').addEventListener('click', () => {
            this.hideAOIInfo();
        });

        // Sidebar toggle
        document.getElementById('sidebarToggle').addEventListener('click', () => {
            this.toggleSidebar();
        });
    }

    setActiveDrawingTool(toolId) {
        // Remove active class from all tools
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Add active class to current tool
        document.getElementById(toolId).classList.add('active');
    }

    toggleLayer(layerName, visible) {
        if (visible) {
            if (!this.map.hasLayer(this.markers[layerName])) {
                this.map.addLayer(this.markers[layerName]);
            }
        } else {
            this.map.removeLayer(this.markers[layerName]);
        }
    }

    async runAnalysis() {
        if (!this.currentAOI || this.isAnalyzing) return;

        this.isAnalyzing = true;
        this.showLoadingOverlay();
        this.updateAnalysisStatus('analyzing', 'Analyzing...');

        try {
            // Simulate AI analysis delay
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Calculate analysis results
            const results = this.calculateAnalysisResults();
            
            // Display results
            this.displayAnalysisResults(results);
            this.updateAnalysisStatus('complete', 'Analysis Complete');
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.updateAnalysisStatus('error', 'Analysis Failed');
        } finally {
            this.isAnalyzing = false;
            this.hideLoadingOverlay();
        }
    }

    calculateAnalysisResults() {
        // Get analysis parameters
        const params = {
            renewable: parseInt(document.getElementById('renewableWeight').value) / 100,
            demand: parseInt(document.getElementById('demandWeight').value) / 100,
            infrastructure: parseInt(document.getElementById('infraWeight').value) / 100,
            economic: parseInt(document.getElementById('economicWeight').value) / 100
        };

        // Simulate AI analysis with realistic results
        const scores = {
            overall: 0.75 + Math.random() * 0.2,
            renewable: 0.70 + Math.random() * 0.25,
            demand: 0.65 + Math.random() * 0.3,
            infrastructure: 0.80 + Math.random() * 0.15,
            economic: 0.60 + Math.random() * 0.35
        };

        // Calculate weighted overall score
        scores.overall = (
            scores.renewable * params.renewable +
            scores.demand * params.demand +
            scores.infrastructure * params.infrastructure +
            scores.economic * params.economic
        );

        return {
            scores,
            recommendations: this.generateRecommendations(scores),
            economics: this.calculateEconomics(),
            risks: this.assessRisks(scores)
        };
    }

    generateRecommendations(scores) {
        const recommendations = [];
        
        if (scores.renewable > 0.8) {
            recommendations.push('Excellent renewable energy potential in this area');
        }
        if (scores.demand > 0.7) {
            recommendations.push('Strong hydrogen demand within reasonable distance');
        }
        if (scores.infrastructure < 0.6) {
            recommendations.push('Infrastructure development required for optimal operations');
        }
        if (scores.economic > 0.7) {
            recommendations.push('Favorable economic conditions for hydrogen development');
        }
        
        return recommendations;
    }

    calculateEconomics() {
        return {
            capex: (200 + Math.random() * 150) * 1000000, // $200-350M
            opex: (15 + Math.random() * 10) * 1000000, // $15-25M annually
            roi: 12 + Math.random() * 8, // 12-20%
            payback: 6 + Math.random() * 4 // 6-10 years
        };
    }

    assessRisks(scores) {
        return [
            { factor: 'Technical Risk', level: scores.infrastructure > 0.7 ? 'Low' : 'Medium' },
            { factor: 'Market Risk', level: scores.demand > 0.7 ? 'Low' : 'High' },
            { factor: 'Regulatory Risk', level: 'Medium' },
            { factor: 'Environmental Risk', level: scores.renewable > 0.8 ? 'Low' : 'Medium' }
        ];
    }

    displayAnalysisResults(results) {
        const content = document.getElementById('analysisPanelContent');
        
        content.innerHTML = `
            <div class="analysis-results">
                <div class="analysis-section">
                    <h4>Overall Assessment</h4>
                    <div class="score-display">
                        <div class="score-circle" style="background: conic-gradient(var(--color-primary) ${results.scores.overall * 360}deg, var(--color-secondary) 0deg)">
                            ${Math.round(results.scores.overall * 100)}
                        </div>
                        <div class="score-details">
                            <p class="score-title">Site Suitability Score</p>
                            <h3 class="score-value">${Math.round(results.scores.overall * 100)}%</h3>
                        </div>
                    </div>
                </div>

                <div class="analysis-section">
                    <h4>Detailed Scores</h4>
                    <div class="chart-container" style="position: relative; height: 250px;">
                        <canvas id="scoresChart"></canvas>
                    </div>
                </div>

                <div class="analysis-section">
                    <h4>Economic Analysis</h4>
                    <div class="economic-metrics">
                        <div class="metric-row">
                            <span>CAPEX:</span>
                            <span>$${(results.economics.capex / 1000000).toFixed(0)}M</span>
                        </div>
                        <div class="metric-row">
                            <span>Annual OPEX:</span>
                            <span>$${(results.economics.opex / 1000000).toFixed(0)}M</span>
                        </div>
                        <div class="metric-row">
                            <span>Expected ROI:</span>
                            <span>${results.economics.roi.toFixed(1)}%</span>
                        </div>
                        <div class="metric-row">
                            <span>Payback Period:</span>
                            <span>${results.economics.payback.toFixed(1)} years</span>
                        </div>
                    </div>
                </div>

                <div class="analysis-section">
                    <h4>Risk Assessment</h4>
                    <div class="risk-list">
                        ${results.risks.map(risk => `
                            <div class="risk-item">
                                <span>${risk.factor}</span>
                                <span class="risk-level risk-${risk.level.toLowerCase()}">${risk.level}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="analysis-section">
                    <h4>Key Recommendations</h4>
                    <ul class="recommendations-list">
                        ${results.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;

        // Create scores chart
        this.createScoresChart(results.scores);
        
        // Add custom styles
        const style = document.createElement('style');
        style.textContent = `
            .metric-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: var(--space-8);
                font-size: var(--font-size-sm);
            }
            .risk-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: var(--space-8);
                font-size: var(--font-size-sm);
            }
            .risk-level {
                padding: var(--space-4) var(--space-8);
                border-radius: var(--radius-sm);
                font-weight: var(--font-weight-medium);
                font-size: var(--font-size-xs);
            }
            .risk-low { background: var(--color-bg-3); color: var(--color-success); }
            .risk-medium { background: var(--color-bg-2); color: var(--color-warning); }
            .risk-high { background: var(--color-bg-4); color: var(--color-error); }
            .recommendations-list {
                margin: 0;
                padding-left: var(--space-20);
            }
            .recommendations-list li {
                margin-bottom: var(--space-8);
                font-size: var(--font-size-sm);
                line-height: 1.4;
            }
        `;
        document.head.appendChild(style);
    }

    createScoresChart(scores) {
        const ctx = document.getElementById('scoresChart');
        if (this.analysisCharts.scores) {
            this.analysisCharts.scores.destroy();
        }

        this.analysisCharts.scores = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Renewable Access', 'Demand Proximity', 'Infrastructure', 'Economic Factors'],
                datasets: [{
                    label: 'Site Score',
                    data: [
                        Math.round(scores.renewable * 100),
                        Math.round(scores.demand * 100),
                        Math.round(scores.infrastructure * 100),
                        Math.round(scores.economic * 100)
                    ],
                    borderColor: '#32808D',
                    backgroundColor: 'rgba(50, 128, 141, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: '#32808D',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    clearAnalysis() {
        const content = document.getElementById('analysisPanelContent');
        content.innerHTML = `
            <div class="analysis-placeholder">
                <div class="placeholder-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                        <polyline points="22,12 18,12 15,21 9,3 6,12 2,12"></polyline>
                    </svg>
                </div>
                <h4>Draw an AOI to begin analysis</h4>
                <p>Use the drawing tools to select an area of interest for hydrogen infrastructure analysis.</p>
            </div>
        `;
        this.updateAnalysisStatus('ready', 'Ready');
    }

    updateAnalysisStatus(status, text) {
        const indicator = document.getElementById('analysisStatus');
        const statusText = document.getElementById('analysisStatusText');
        
        indicator.className = `status-indicator ${status}`;
        statusText.textContent = text;
    }

    showLoadingOverlay() {
        document.getElementById('loadingOverlay').classList.add('active');
    }

    hideLoadingOverlay() {
        document.getElementById('loadingOverlay').classList.remove('active');
    }

    exportAnalysisData() {
        if (!this.currentAOI) {
            this.showNotification('No analysis data to export', 'warning');
            return;
        }

        const data = {
            timestamp: new Date().toISOString(),
            aoi: this.currentAOI.toGeoJSON(),
            analysis: 'Analysis results would be included here',
            platform_version: this.data.platform_metadata.version
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hydrogen-analysis-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    setupThemeToggle() {
        const themeToggle = document.getElementById('themeToggle');
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-color-scheme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-color-scheme', newTheme);
        });
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const mainApp = document.querySelector('.main-app');
        
        sidebar.classList.toggle('collapsed');
        mainApp.classList.toggle('sidebar-collapsed');
    }

    showNotification(message, type = 'info') {
        // Simple notification system
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 90px;
            right: 20px;
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-base);
            padding: var(--space-12) var(--space-16);
            box-shadow: var(--shadow-lg);
            z-index: 10000;
            animation: slideIn 0.3s ease-out;
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    initializeUI() {
        // Set initial analysis status
        this.updateAnalysisStatus('ready', 'Ready');
        
        // Initialize parameter values
        document.querySelectorAll('.parameter-slider').forEach(slider => {
            const valueSpan = slider.parentNode.querySelector('.parameter-value');
            valueSpan.textContent = slider.value + '%';
        });

        console.log('UI initialized successfully');
    }
}

// Initialize the platform when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.hydrogenPlatform = new HydrogenIntelligencePlatform();
});

// Add CSS animation for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);