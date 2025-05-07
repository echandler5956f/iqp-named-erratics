import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom'; // Keep createPortal
import styles from './AddEditFilterModal.module.css';

// Portal requires access to the DOM, ensure this runs client-side
const modalRoot = document.getElementById('modal-root');

function AddEditFilterModal({ isOpen, onClose, onSave, existingFilter, filterDefinitions, configComponentStyles }) {
  // Restore state variables
  const [filterName, setFilterName] = useState('');
  const [filterType, setFilterType] = useState(''); // Initialize empty, useEffect will set default
  const [config, setConfig] = useState({});

  useEffect(() => {
    if (existingFilter) {
      // Editing existing filter
      setFilterName(existingFilter.name);
      setFilterType(existingFilter.type);
      setConfig(existingFilter.config || {}); // Ensure config is at least an empty object
    } else {
      // Adding new filter: determine initial type
      const initialType = filterDefinitions && Object.keys(filterDefinitions).length > 0 
                          ? Object.keys(filterDefinitions)[0] 
                          : '';
      setFilterName(''); // Start with blank name
      setFilterType(initialType);
      if (initialType && filterDefinitions[initialType]) {
        // Set default config for the initial type
        setConfig(filterDefinitions[initialType].defaultConfig || {}); 
      } else {
        setConfig({});
      }
    }
  }, [existingFilter, filterDefinitions]);

  // Effect to update config when filterType changes for a *new* filter
  useEffect(() => {
    if (!existingFilter && filterType && filterDefinitions[filterType]) {
      setConfig(filterDefinitions[filterType].defaultConfig || {});
    }
    // Do not reset config if editing, type changes should be handled carefully if allowed when editing
  }, [filterType, existingFilter, filterDefinitions]); 

  // --- ESC Key Handler ---
  useEffect(() => {
    const handleEsc = (event) => {
       if (event.key === 'Escape') {
          onClose();
       }
    };
    if (isOpen) {
        document.addEventListener('keydown', handleEsc);
    }
    // Cleanup listener
    return () => {
      document.removeEventListener('keydown', handleEsc);
    };
  }, [isOpen, onClose]);
  // --- End ESC Key Handler ---

  const handleTypeChange = (e) => {
    const newType = e.target.value;
    setFilterType(newType);
    // Config update for new type is handled by the useEffect above
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Restore name validation
    if (!filterName.trim()) {
      alert('Filter name cannot be empty.');
      return;
    }
    if (!filterType || !filterDefinitions[filterType]) {
      alert('Invalid filter type selected.');
      return;
    }
    onSave({ name: filterName, type: filterType, config });
  };

  if (!isOpen || !modalRoot) {
    return null;
  }

  const CurrentFilterConfigComponent = filterDefinitions && filterType && filterDefinitions[filterType]?.Component;

  // Restore original modal content JSX, wrapped in the portal
  const modalContent = (
    <div className={styles.modalOverlay} onClick={onClose}> 
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}> 
        <h3>{existingFilter ? 'Edit Filter' : 'Add New Filter'}</h3>
        <form onSubmit={handleSubmit}>
          <div className={styles.formGroup}>
            <label htmlFor="filterName">Filter Name:</label>
            <input
              type="text"
              id="filterName"
              value={filterName}
              onChange={(e) => setFilterName(e.target.value)}
              required
              placeholder="e.g., Large Erratics, Near Rivers"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="filterType">Filter Type:</label>
            <select id="filterType" value={filterType} onChange={handleTypeChange} required disabled={!!existingFilter}>
              <option value="" disabled={Object.keys(filterDefinitions || {}).length > 0}>Select a type...</option>
              {filterDefinitions && Object.entries(filterDefinitions).map(([typeKey, def]) => (
                <option key={typeKey} value={typeKey}>
                  {def.label}
                </option>
              ))}
            </select>
          </div>

          {CurrentFilterConfigComponent && (
            <div className={styles.formGroup}>
              <label>Configuration:</label>
              {/* Pass the configComponentStyles (FilterPanel.module.css) */}
              <div className={styles.configurationArea}> {/* Optional wrapper for styling */} 
                <CurrentFilterConfigComponent 
                  config={config} 
                  onChange={setConfig} 
                  styles={configComponentStyles} 
                />
              </div>
            </div>
          )}

          <div className={styles.modalActions}>
            <button type="submit" className={styles.saveButton}>Save Filter</button>
            <button type="button" onClick={onClose} className={styles.cancelButton}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );

  return ReactDOM.createPortal(
    modalContent,
    modalRoot
  );
}

export default AddEditFilterModal; 