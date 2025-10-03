// CyberGuard Predictor - Main JavaScript

console.log('🛡️ CyberGuard Predictor loaded successfully');

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing CyberGuard dashboard...');
    
    // Add any additional initialization here
    if (typeof initializeMap !== 'undefined') {
        console.log('📍 Map initialization available');
    }
    
    if (typeof initializeWebSocket !== 'undefined') {
        console.log('🔗 WebSocket initialization available');
    }
    
    console.log('✅ CyberGuard dashboard ready');
});

// Utility functions for the dashboard
window.CyberGuard = {
    // Format numbers with commas
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },
    
    // Format risk level with appropriate color
    formatRiskLevel: function(level) {
        const colors = {
            'HIGH': 'danger',
            'MEDIUM': 'warning', 
            'LOW': 'success'
        };
        return `<span class="badge bg-${colors[level] || 'secondary'}">${level}</span>`;
    },
    
    // Show notification
    showNotification: function(message, type = 'info') {
        console.log(`📢 ${type.toUpperCase()}: ${message}`);
        // Could integrate with toast notifications later
    }
};

console.log('📦 CyberGuard utilities loaded');