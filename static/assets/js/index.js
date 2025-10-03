// CyberGuard Predictor - Index Page Scripts

console.log('🏠 CyberGuard index page scripts loaded');

// Enhanced dashboard functionality
(function() {
    'use strict';
    
    // Dashboard configuration
    const DASHBOARD_CONFIG = {
        refreshInterval: 30000, // 30 seconds
        mapCenter: [20.5937, 78.9629], // Center of India
        mapZoom: 5,
        alertLimit: 10
    };
    
    // Enhanced map functionality
    window.enhanceMap = function(map) {
        if (!map) return;
        
        // Add India boundary (simplified)
        const indiaBounds = [
            [6.4627, 68.1097],  // Southwest
            [35.5044, 97.3954]  // Northeast
        ];
        
        // Add a rectangle showing India bounds
        L.rectangle(indiaBounds, {
            color: '#1e40af',
            weight: 2,
            fillOpacity: 0.1
        }).addTo(map);
        
        console.log('🗺️ Enhanced map with India boundaries');
    };
    
    // Enhanced WebSocket handling
    window.enhanceWebSocket = function() {
        // Add reconnection logic
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function attemptReconnect() {
            if (reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                console.log(`🔄 WebSocket reconnection attempt ${reconnectAttempts}`);
                setTimeout(() => {
                    if (typeof initializeWebSocket === 'function') {
                        initializeWebSocket();
                    }
                }, 2000 * reconnectAttempts);
            }
        }
        
        return { attemptReconnect };
    };
    
    // Performance monitoring
    window.performance && window.performance.mark && window.performance.mark('cyberguard-index-loaded');
    
    console.log('✅ CyberGuard index enhancements ready');
})();