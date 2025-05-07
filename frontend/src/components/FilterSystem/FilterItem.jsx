import React from 'react';
import styles from './FilterItem.module.css';

// Basic toggle switch component (can be replaced with a fancier one from a library)
const ToggleSwitch = ({ id, checked, onChange, label }) => (
  <label htmlFor={id} className={styles.toggleSwitchContainer}>
    <input
      type="checkbox"
      id={id}
      checked={checked}
      onChange={onChange}
      className={styles.toggleInput}
    />
    <span className={styles.toggleSlider}></span>
    {label && <span className={styles.toggleLabel}>{label}</span>}
  </label>
);

function FilterItem({ filter, filterDefinition, onToggleActive, onEdit, onDelete }) {
  if (!filterDefinition) {
    console.warn(`No filter definition found for type: ${filter.type}`);
    return <div className={styles.filterItemError}>Error: Unknown filter type '{filter.type}'</div>;
  }

  let configSummary = '';
  if (filter.type === 'size') {
    const { min, max } = filter.config;
    const minStr = (min !== null && !isNaN(min)) ? `>= ${min} (meters)` : '';
    const maxStr = (max !== null && !isNaN(max)) ? `<= ${max} (meters)` : '';
    configSummary = `Size: ${minStr}${minStr && maxStr ? ' & ' : ''}${maxStr}`.trim() || 'Any size';
    if (!minStr && !maxStr && (min !== null || max !== null)) configSummary = 'Invalid size range'; // Handle case where input might be non-numeric

  } else if (filter.type === 'proximity_water') {
    const { maxDist } = filter.config;
    configSummary = (maxDist !== null && !isNaN(maxDist)) ? `Dist. to Water <= ${maxDist} (meters)` : 'Any distance to water';

  } else if (filter.type === 'rock_type') {
    configSummary = filter.config.type ? `Rock Type: ${filter.config.type}` : 'Any rock type';
  
  } else if (filter.type === 'usage_type') {
    configSummary = filter.config.tag ? `Usage Contains: ${filter.config.tag}` : 'Any usage type';

  } else if (filter.type === 'has_inscriptions') {
    configSummary = filter.config.required ? 'Must Have Inscriptions' : 'Inscriptions not required';
  }

  return (
    <div className={`${styles.filterItem} ${filter.isActive ? styles.active : styles.inactive}`}>
      <div className={styles.filterHeader}>
        <span className={styles.filterName}>{filter.name}</span>
        <span className={styles.filterType}>({filterDefinition.label})</span>
      </div>
      {configSummary && <p className={styles.filterSummary}>{configSummary}</p>}
      <div className={styles.filterControls}>
        <ToggleSwitch
          id={`toggle-${filter.id}`}
          checked={filter.isActive}
          onChange={onToggleActive}
          label={filter.isActive ? 'Active' : 'Inactive'}
        />
        <button 
          onClick={onEdit} 
          className={`${styles.controlButton} ${styles.editButton}`}
          title="Edit filter name and settings"
        >
          Edit
        </button>
        <button 
          onClick={onDelete} 
          className={`${styles.controlButton} ${styles.deleteButton}`}
          title="Delete this filter"
        >
          Delete
        </button>
      </div>
    </div>
  );
}

export default FilterItem; 