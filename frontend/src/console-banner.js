/**
 * DENALI AI Console Banner
 * Displays branded console message on page load (runs once)
 */
(function() {
    'use strict';

    // Check if banner was already displayed in this session
    if (window.__DENALI_BANNER_DISPLAYED__) {
        return;
    }

    // Mark as displayed
    window.__DENALI_BANNER_DISPLAYED__ = true;

    // ASCII Art Banner
    const banner = `
██████╗ ███████╗███╗   ██╗ █████╗ ██╗     ██╗     █████╗ ██╗
██╔══██╗██╔════╝████╗  ██║██╔══██╗██║     ██║    ██╔══██╗██║
██║  ██║█████╗  ██╔██╗ ██║███████║██║     ██║    ███████║██║
██║  ██║██╔══╝  ██║╚██╗██║██╔══██║██║     ██║    ██╔══██║██║
██████╔╝███████╗██║ ╚████║██║  ██║███████╗██████╗██║  ██║██║
╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═════╝╚═╝  ╚═╝╚═╝
    `;

    // Styles for different parts
    const styles = {
        banner: 'color: #00d4ff; font-weight: bold; font-size: 12px; line-height: 1.2; font-family: monospace;',
        title: 'color: #00d4ff; font-weight: bold; font-size: 24px; font-family: system-ui, -apple-system, sans-serif;',
        subtitle: 'color: #666; font-size: 14px; font-family: system-ui, -apple-system, sans-serif;',
        info: 'color: #999; font-size: 12px; font-family: system-ui, -apple-system, sans-serif;',
        link: 'color: #00d4ff; font-size: 12px; font-family: system-ui, -apple-system, sans-serif; text-decoration: underline;',
        divider: 'color: #333; font-size: 12px;'
    };

    // Display banner
    console.log('%c' + banner, styles.banner);
    console.log('%cRELO Classifier', styles.title);
    console.log('%cIntelligent Returns Classification System', styles.subtitle);
    console.log('%c' + '─'.repeat(60), styles.divider);
    console.log('%cVersion: 1.0.0', styles.info);
    console.log('%cEnvironment: ' + (location.hostname === 'localhost' ? 'Development' : 'Production'), styles.info);
    console.log('%c' + '─'.repeat(60), styles.divider);
    console.log('%c  This browser console is intended for developers only.', styles.info);
    console.log('%cDo not paste code here unless you understand what it does.', styles.info);
    console.log('%c' + '─'.repeat(60), styles.divider);
    console.log(' ');
})();
