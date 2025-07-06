document.addEventListener('DOMContentLoaded', () => {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    const currentPath = window.location.pathname.split('/').pop();

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }

    // Function to set active link
    function setActiveLink() {
        // Desktop navigation links
        document.querySelectorAll('nav div.hidden.md\\:flex a').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('font-bold', 'text-blue-600');
                link.classList.remove('text-slate-700');
            }
        });

        // Mobile navigation links
        document.querySelectorAll('#mobile-menu a').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('font-bold', 'bg-slate-100', 'text-blue-600');
                link.classList.remove('text-slate-700', 'hover:bg-slate-100');
            }
        });
    }

    setActiveLink();
});