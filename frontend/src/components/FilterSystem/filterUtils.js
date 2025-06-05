export const passesAllFilters = (erratic, activeFilters, filterDefinitions) => {
  if (!activeFilters || activeFilters.length === 0) {
    return true;
  }

  for (const filter of activeFilters) {
    if (!filter.isActive) continue;

    const definition = filterDefinitions[filter.type];
    if (!definition) {
      console.warn(`No definition for filter type: ${filter.type}`);
      continue;
    }

    let passesCurrentFilter = true; // Default to true, only fail if specific check fails
    switch (filter.type) {
      case 'size':
        const minSize = filter.config.min;
        const maxSize = filter.config.max;
        const erraticSize = erratic.size_meters;

        if (erraticSize === null || erraticSize === undefined) {
          passesCurrentFilter = false;
          break;
        }

        if (minSize !== null && typeof minSize === 'number' && !isNaN(minSize)) {
          if (erraticSize < minSize) {
            passesCurrentFilter = false;
            break;
          }
        }

        if (maxSize !== null && typeof maxSize === 'number' && !isNaN(maxSize)) {
          if (erraticSize > maxSize) {
            passesCurrentFilter = false;
          }
        }
        break;
        
      case 'proximity_water':
        const maxDistWater = filter.config.maxDist;
        const erraticDistWater = erratic.nearest_water_body_dist;

        if (erraticDistWater === null || erraticDistWater === undefined) {
           passesCurrentFilter = false;
           break;
        }
        
        if (maxDistWater !== null && typeof maxDistWater === 'number' && !isNaN(maxDistWater)) {
          if (erraticDistWater > maxDistWater) {
            passesCurrentFilter = false;
          }
        }
        break;

      case 'proximity_forest_trail':
        const maxDistForestTrail = filter.config.maxDist;
        const erraticDistForestTrail = erratic.nearest_forest_trail_dist;

        if (erraticDistForestTrail === null || erraticDistForestTrail === undefined) {
          passesCurrentFilter = false;
          break;
        }
        
        if (maxDistForestTrail !== null && typeof maxDistForestTrail === 'number' && !isNaN(maxDistForestTrail)) {
          if (erraticDistForestTrail > maxDistForestTrail) {
            passesCurrentFilter = false;
          }
        }
        break;

      case 'proximity_settlement':
        const maxDistSettlement = filter.config.maxDist;
        const erraticDistSettlement = erratic.nearest_settlement_dist;

        if (erraticDistSettlement === null || erraticDistSettlement === undefined) {
          passesCurrentFilter = false;
          break;
        }
        
        if (maxDistSettlement !== null && typeof maxDistSettlement === 'number' && !isNaN(maxDistSettlement)) {
          if (erraticDistSettlement > maxDistSettlement) {
            passesCurrentFilter = false;
          }
        }
        break;

      case 'proximity_road':
        const maxDistRoad = filter.config.maxDist;
        const erraticDistRoad = erratic.nearest_road_dist;

        if (erraticDistRoad === null || erraticDistRoad === undefined) {
          passesCurrentFilter = false;
          break;
        }
        
        if (maxDistRoad !== null && typeof maxDistRoad === 'number' && !isNaN(maxDistRoad)) {
          if (erraticDistRoad > maxDistRoad) {
            passesCurrentFilter = false;
          }
        }
        break;

      case 'elevation_category':
        const filterElevationCategory = filter.config.category?.trim().toLowerCase();
        if (filterElevationCategory) {
          if (!erratic.elevation_category || erratic.elevation_category.toLowerCase() !== filterElevationCategory) {
            passesCurrentFilter = false;
          }
        }
        break;

      case 'rock_type':
        const filterRockType = filter.config.type?.trim().toLowerCase();
        if (filterRockType) { 
          if (!erratic.rock_type || erratic.rock_type.toLowerCase() !== filterRockType) {
            passesCurrentFilter = false;
          }
        } 
        break;

      case 'usage_type':
        const filterUsageTag = filter.config.tag?.trim().toLowerCase();
        if (filterUsageTag) {
          const usageTags = erratic.usage_type;
          if (!Array.isArray(usageTags) || !usageTags.some(tag => tag.toLowerCase() === filterUsageTag)) {
            passesCurrentFilter = false; 
          }
        }
        break;

      case 'has_inscriptions':
        if (filter.config.required === true) { 
          if (erratic.has_inscriptions !== true) {
            passesCurrentFilter = false;
          }
        }
        break;

      case 'accessibility_score':
        const minAccessibility = filter.config.min;
        const maxAccessibility = filter.config.max;
        const erraticAccessibility = erratic.accessibility_score;

        if (erraticAccessibility === null || erraticAccessibility === undefined) {
          passesCurrentFilter = false;
          break;
        }
        if (minAccessibility !== null && typeof minAccessibility === 'number' && !isNaN(minAccessibility)) {
          if (erraticAccessibility < minAccessibility) {
            passesCurrentFilter = false;
            break;
          }
        }
        if (maxAccessibility !== null && typeof maxAccessibility === 'number' && !isNaN(maxAccessibility)) {
          if (erraticAccessibility > maxAccessibility) {
            passesCurrentFilter = false;
          }
        }
        break;

      case 'terrain_landform':
        const filterLandformType = filter.config.type?.trim().toLowerCase();
        if (filterLandformType) {
          if (!erratic.terrain_landform || erratic.terrain_landform.toLowerCase() !== filterLandformType) {
            passesCurrentFilter = false;
          }
        }
        break;

      default:
        console.warn(`Unknown filter type in passesAllFilters: ${filter.type}`);
        break;
    }

    if (!passesCurrentFilter) {
      return false;
    }
  }

  return true;
};

export const getVisibleErraticIds = (allErraticData, activeFilters, filterDefinitions) => {
  if (!activeFilters || activeFilters.length === 0) {
    return new Set(allErraticData.map(e => e.id));
  }
  
  const visibleIds = new Set();
  allErraticData.forEach(erratic => {
    if (passesAllFilters(erratic, activeFilters, filterDefinitions)) {
      visibleIds.add(erratic.id);
    }
  });
  return visibleIds;
}; 