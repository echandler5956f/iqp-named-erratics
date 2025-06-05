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
  switch (filter.type) {
    case 'size':
      const { min: sizeMin, max: sizeMax } = filter.config;
      const minSizeStr = (sizeMin !== null && !isNaN(sizeMin)) ? `>= ${sizeMin}m` : '';
      const maxSizeStr = (sizeMax !== null && !isNaN(sizeMax)) ? `<= ${sizeMax}m` : '';
      configSummary = `Size: ${minSizeStr}${minSizeStr && maxSizeStr ? ' & ' : ''}${maxSizeStr}`.trim() || 'Any size';
      break;

    case 'proximity_water':
      const { maxDist: waterMaxDist } = filter.config;
      configSummary = (waterMaxDist !== null && !isNaN(waterMaxDist)) ? `Water <= ${waterMaxDist}m` : 'Any distance to water';
      break;

    case 'proximity_forest_trail':
      const { maxDist: trailMaxDist } = filter.config;
      configSummary = (trailMaxDist !== null && !isNaN(trailMaxDist)) ? `Forest Trail <= ${trailMaxDist}m` : 'Any distance to forest trail';
      break;

    case 'proximity_settlement':
      const { maxDist: settlementMaxDist } = filter.config;
      configSummary = (settlementMaxDist !== null && !isNaN(settlementMaxDist)) ? `Settlement <= ${settlementMaxDist}m` : 'Any distance to settlement';
      break;

    case 'proximity_road':
      const { maxDist: roadMaxDist } = filter.config;
      configSummary = (roadMaxDist !== null && !isNaN(roadMaxDist)) ? `Road <= ${roadMaxDist}m` : 'Any distance to road';
      break;

    case 'proximity_natd_road':
      const { maxDist: natdRoadMaxDist } = filter.config;
      configSummary = (natdRoadMaxDist !== null && !isNaN(natdRoadMaxDist)) ? `NATD Road <= ${natdRoadMaxDist}m` : 'Any distance to NATD road';
      break;

    case 'proximity_native_territory':
      const { maxDist: territoryMaxDist } = filter.config;
      configSummary = (territoryMaxDist !== null && !isNaN(territoryMaxDist)) ? `Native Territory <= ${territoryMaxDist}m` : 'Any distance to native territory';
      break;

    case 'elevation_category':
      configSummary = filter.config.category ? `Elevation: ${filter.config.category}` : 'Any elevation category';
      break;

    case 'elevation':
      const { min: elevMin, max: elevMax } = filter.config;
      const minElevStr = (elevMin !== null && !isNaN(elevMin)) ? `>= ${elevMin}m` : '';
      const maxElevStr = (elevMax !== null && !isNaN(elevMax)) ? `<= ${elevMax}m` : '';
      configSummary = `Elevation: ${minElevStr}${minElevStr && maxElevStr ? ' & ' : ''}${maxElevStr}`.trim() || 'Any elevation';
      break;

    case 'rock_type':
      configSummary = filter.config.type ? `Rock Type: ${filter.config.type}` : 'Any rock type';
      break;

    case 'estimated_age':
      configSummary = filter.config.age ? `Age: ${filter.config.age}` : 'Any estimated age';
      break;

    case 'usage_type':
      configSummary = filter.config.tag ? `Usage: ${filter.config.tag}` : 'Any usage type';
      break;

    case 'has_inscriptions':
      configSummary = filter.config.required ? 'Must have inscriptions' : 'Inscriptions not required';
      break;

    case 'accessibility_score':
      const { min: accessMin, max: accessMax } = filter.config;
      const minAccessStr = (accessMin !== null && !isNaN(accessMin)) ? `>= ${accessMin}` : '';
      const maxAccessStr = (accessMax !== null && !isNaN(accessMax)) ? `<= ${accessMax}` : '';
      configSummary = `Accessibility: ${minAccessStr}${minAccessStr && maxAccessStr ? ' & ' : ''}${maxAccessStr}`.trim() || 'Any accessibility';
      break;

    case 'cultural_significance_score':
      const { min: cultMin, max: cultMax } = filter.config;
      const minCultStr = (cultMin !== null && !isNaN(cultMin)) ? `>= ${cultMin}` : '';
      const maxCultStr = (cultMax !== null && !isNaN(cultMax)) ? `<= ${cultMax}` : '';
      configSummary = `Cultural Significance: ${minCultStr}${minCultStr && maxCultStr ? ' & ' : ''}${maxCultStr}`.trim() || 'Any cultural significance';
      break;

    case 'terrain_landform':
      configSummary = filter.config.type ? `Landform: ${filter.config.type}` : 'Any terrain landform';
      break;

    case 'slope_position':
      configSummary = filter.config.type ? `Slope Position: ${filter.config.type}` : 'Any slope position';
      break;

    case 'size_category':
      configSummary = filter.config.category ? `Size Category: ${filter.config.category}` : 'Any size category';
      break;

    case 'geological_type':
      configSummary = filter.config.type ? `Geological Type: ${filter.config.type}` : 'Any geological type';
      break;

    case 'displacement_distance':
      const { min: displMin, max: displMax } = filter.config;
      const minDisplStr = (displMin !== null && !isNaN(displMin)) ? `>= ${displMin}m` : '';
      const maxDisplStr = (displMax !== null && !isNaN(displMax)) ? `<= ${displMax}m` : '';
      configSummary = `Displacement: ${minDisplStr}${minDisplStr && maxDisplStr ? ' & ' : ''}${maxDisplStr}`.trim() || 'Any displacement';
      break;

    case 'ruggedness':
      const { min: rugMin, max: rugMax } = filter.config;
      const minRugStr = (rugMin !== null && !isNaN(rugMin)) ? `>= ${rugMin}` : '';
      const maxRugStr = (rugMax !== null && !isNaN(rugMax)) ? `<= ${rugMax}` : '';
      configSummary = `Ruggedness (TRI): ${minRugStr}${minRugStr && maxRugStr ? ' & ' : ''}${maxRugStr}`.trim() || 'Any ruggedness';
      break;

    case 'discovery_date':
      const { startYear, endYear } = filter.config;
      const startYearStr = (startYear !== null && !isNaN(startYear)) ? `>= ${startYear}` : '';
      const endYearStr = (endYear !== null && !isNaN(endYear)) ? `<= ${endYear}` : '';
      configSummary = `Discovery: ${startYearStr}${startYearStr && endYearStr ? ' & ' : ''}${endYearStr}`.trim() || 'Any discovery date';
      break;

    default:
      configSummary = 'Filter configuration';
      break;
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