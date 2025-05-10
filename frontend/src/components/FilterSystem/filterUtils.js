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
        const minSize = filter.config.min; // Keep null if not set
        const maxSize = filter.config.max; // Keep null if not set
        const erraticSize = erratic.size_meters;

        if (erraticSize === null || erraticSize === undefined) {
          passesCurrentFilter = false; // Cannot pass size filter if erratic has no size
          break;
        }

        // Check min bound only if minSize is a valid number
        if (minSize !== null && typeof minSize === 'number' && !isNaN(minSize)) {
          if (erraticSize < minSize) {
            passesCurrentFilter = false;
            break; // Exit case early if fails
          }
        }

        // Check max bound only if maxSize is a valid number
        if (maxSize !== null && typeof maxSize === 'number' && !isNaN(maxSize)) {
          if (erraticSize > maxSize) {
            passesCurrentFilter = false;
            // break; // No need to break, it's the last check for this case
          }
        }
        // If neither check failed, passesCurrentFilter remains true
        break;
        
      case 'proximity_water':
        const maxDist = filter.config.maxDist; // Keep null if not set
        const erraticDist = erratic.nearest_water_body_dist;

        // --- DEBUG LOG --- 
        // console.log(`Checking ${erratic.name} (ID: ${erratic.id}) for water proximity: maxDist=${maxDist}, erraticDist=${erraticDist}`);
        // --- END DEBUG LOG ---

        if (erraticDist === null || erraticDist === undefined) {
           // console.log(`-> Failing ${erratic.name} due to missing distance data.`);
           passesCurrentFilter = false; // Cannot pass proximity filter if erratic has no distance data
           break;
        }
        
        // Check max distance only if maxDist is a valid number
        if (maxDist !== null && typeof maxDist === 'number' && !isNaN(maxDist)) {
          if (erraticDist > maxDist) {
            // console.log(`-> Failing ${erratic.name} because ${erraticDist} > ${maxDist}`);
            passesCurrentFilter = false;
          } else {
            // console.log(`-> Passing ${erratic.name} because ${erraticDist} <= ${maxDist}`);
          }
        } else {
           // console.log(`-> Passing ${erratic.name} because no maxDist limit set.`);
        }
        // If check didn't fail (or wasn't applied), passesCurrentFilter remains true
        break;

      // --- NEW FILTER LOGIC ---
      case 'rock_type':
        const filterRockType = filter.config.type?.trim().toLowerCase();
        // Only filter if type is specified, case-insensitive compare
        if (filterRockType) { 
          if (!erratic.rock_type || erratic.rock_type.toLowerCase() !== filterRockType) {
            passesCurrentFilter = false;
          }
        } 
        // If filterRockType is empty/null, this filter doesn't exclude anything
        break;

      case 'usage_type':
        const filterUsageTag = filter.config.tag?.trim().toLowerCase();
        // Only filter if tag is specified, check if array includes tag (case-insensitive)
        if (filterUsageTag) {
          const usageTags = erratic.usage_type; // This is expected to be an array of strings
          if (!Array.isArray(usageTags) || !usageTags.some(tag => tag.toLowerCase() === filterUsageTag)) {
            passesCurrentFilter = false; 
          }
        }
        // If filterUsageTag is empty/null, this filter doesn't exclude anything
        break;

      case 'has_inscriptions':
        // Filter is only active if config.required is true (controlled by checkbox)
        if (filter.config.required === true) { 
          if (erratic.has_inscriptions !== true) { // Erratic must have inscriptions
            passesCurrentFilter = false;
          }
        }
        // If config.required is false (checkbox unchecked), this filter doesn't exclude anything
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

      case 'nearest_colonial_road_dist':
        const maxDistColonialRoad = filter.config.maxDist;
        const erraticDistColonialRoad = erratic.nearest_colonial_road_dist;

        if (erraticDistColonialRoad === null || erraticDistColonialRoad === undefined) {
          passesCurrentFilter = false;
          break;
        }
        if (maxDistColonialRoad !== null && typeof maxDistColonialRoad === 'number' && !isNaN(maxDistColonialRoad)) {
          if (erraticDistColonialRoad > maxDistColonialRoad) {
            passesCurrentFilter = false;
          }
        }
        break;

      default:
        console.warn(`Unknown filter type in passesAllFilters: ${filter.type}`);
        // Decide behavior for unknown types - defaulting to pass (true)
        break;
    }

    if (!passesCurrentFilter) {
      return false; // If *any* active filter explicitly fails, the erratic is filtered out
    }
  }

  return true; // Passes all active filters that were applied
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