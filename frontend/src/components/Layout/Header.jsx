import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Header.css';

function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const isActivePath = (path) => {
    return location.pathname === path;
  };

  const closeMenu = () => {
    setIsMenuOpen(false);
  };

  return (
    <header className="header">
      <div className="header__container">
        <div className="header__brand">
          <Link to="/" className="header__logo" onClick={closeMenu}>
            <span className="header__logo-text">
              <span className="header__logo-main">Glacial Erratics</span>
              <span className="header__logo-sub">Research Database</span>
            </span>
          </Link>
        </div>
        
        <button 
          className={`header__mobile-toggle ${isMenuOpen ? 'header__mobile-toggle--open' : ''}`}
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label="Toggle navigation menu"
          aria-expanded={isMenuOpen}
        >
          <span className="header__hamburger"></span>
        </button>
        
        <nav className={`header__nav ${isMenuOpen ? 'header__nav--open' : ''}`}>
          <ul className="header__nav-list">
            <li className="header__nav-item">
              <Link 
                to="/" 
                className={`header__nav-link ${isActivePath('/') ? 'header__nav-link--active' : ''}`}
                onClick={closeMenu}
              >
                Interactive Map
              </Link>
            </li>
            <li className="header__nav-item">
              <Link 
                to="/about" 
                className={`header__nav-link ${isActivePath('/about') ? 'header__nav-link--active' : ''}`}
                onClick={closeMenu}
              >
                About Project
              </Link>
            </li>
            <li className="header__nav-item">
              <Link 
                to="/login" 
                className={`header__nav-link header__nav-link--admin ${isActivePath('/login') ? 'header__nav-link--active' : ''}`}
                onClick={closeMenu}
              >
                Admin Access
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}

export default Header; 