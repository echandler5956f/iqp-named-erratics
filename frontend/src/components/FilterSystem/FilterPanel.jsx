import React, { useState } from 'react';
import styles from './FilterPanel.module.css';
import FilterItem from './FilterItem';
import AddEditFilterModal from './AddEditFilterModal';

// TODO: Replace with actual filter types and configurations
const INITIAL_FILTER_DEFINITIONS = {
  size: {
    label: 'Size (meters)',
    defaultConfig: { min: 0, max: 1000 }, // Example defaults
    Component: ({ config, onChange }) => (
      <div className={styles.filterConfigRow}>
        <label htmlFor="minSize">Min:</label>
        <input
          type="number"
          id="minSize"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: parseFloat(e.target.value) || 0 })}
          placeholder="Min size"
        />
        <label htmlFor="maxSize">Max:</label>
        <input
          type="number"
          id="maxSize"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: parseFloat(e.target.value) || Infinity })}
          placeholder="Max size"
        />
      </div>
    ),
  },
  proximity_water: {
    label: 'Proximity to Water (meters)',
    defaultConfig: { maxDist: 1000 }, // Example default
    Component: ({ config, onChange }) => (
      <div className={styles.filterConfigRow}>
        <label htmlFor="maxDistWater">Max Distance:</label>
        <input
          type="number"
          id="maxDistWater"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: parseFloat(e.target.value) || 0 })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  // Add more filter types here as needed, e.g., rock_type, elevation_category
};

// Helper function modification
const generateDefaultFilterName = (type, existingFilters, definitions) => {
  const baseName = definitions[type]?.label || 'Filter';
  let counter = 0;
  let finalName = '';
  const existingNames = new Set(existingFilters.map(f => f.name));

  // Find the first available number (e.g., "Size Filter 1", "Size Filter 2")
  do {
    counter++;
    finalName = `${baseName} ${counter}`;
  } while (existingNames.has(finalName));

  return finalName;
};

// Now expects filterDefinitions from props
function FilterPanel({ filters, onFiltersChange, filterDefinitions, /* configComponentStyles */ }) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingFilter, setEditingFilter] = useState(null);

  console.log('[FilterPanel] Rendering. isModalOpen:', isModalOpen);

  const handleAddFilter = () => {
    console.log('[FilterPanel] handleAddFilter called');
    setEditingFilter(null);
    setIsModalOpen(true);
  };

  const handleEditFilter = (filterToEdit) => {
    console.log('[FilterPanel] handleEditFilter called for:', filterToEdit?.name);
    setEditingFilter(filterToEdit);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    console.log('[FilterPanel] handleCloseModal called');
    setIsModalOpen(false);
    setEditingFilter(null);
  }

  const handleDeleteFilter = (filterId) => {
    onFiltersChange(filters.filter(f => f.id !== filterId));
  };

  const handleToggleActive = (filterId) => {
    onFiltersChange(
      filters.map(f =>
        f.id === filterId ? { ...f, isActive: !f.isActive } : f
      )
    );
  };

  const handleSaveFilter = (filterData) => {
    console.log('handleSaveFilter received:', filterData);
    if (editingFilter) {
      // EDITING: simplified back - assume name is valid from modal
      onFiltersChange(
        filters.map(f => (f.id === editingFilter.id
            ? { ...f, ...filterData } // Update with all data from modal, keeping original ID/isActive
            : f
        ))
      );
    } else {
      // ADDING: Assume name is provided and validated by modal
      onFiltersChange([
        ...filters,
        { ...filterData, id: Date.now().toString(), isActive: true }, // Assign ID and set active
      ]);
    }
    handleCloseModal();
  };

  // Check if filterDefinitions is available before trying to use it
  if (!filterDefinitions || Object.keys(filterDefinitions).length === 0) {
    console.log('[FilterPanel] No filterDefinitions, rendering minimal panel.');
    return (
      <div className={styles.filterPanel}>
        <h3>Filter Erratics</h3>
        <p className={styles.noFiltersText}>Filter definitions not available.</p>
      </div>
    );
  }

  return (
    <div className={styles.filterPanel}>
      <h3>Filter Erratics</h3>
      <button onClick={handleAddFilter} className={styles.addButton}>
        Add New Filter
      </button>
      <div className={styles.filterList}>
        {filters.length === 0 && <p className={styles.noFiltersText}>No filters applied. Add a filter to refine the map.</p>}
        {filters.map(filter => (
          <FilterItem
            key={filter.id}
            filter={filter}
            filterDefinition={filterDefinitions[filter.type]} // Use passed definitions
            // Pass FilterPanel's styles to FilterItem if it needs to render config components directly,
            // or if config summary needs these styles (unlikely for summary).
            // For now, FilterItem doesn't render the config component directly.
            onToggleActive={() => handleToggleActive(filter.id)}
            onEdit={() => handleEditFilter(filter)}
            onDelete={() => handleDeleteFilter(filter.id)}
          />
        ))}
      </div>
      {/* Explicitly log before rendering AddEditFilterModal */}
      {isModalOpen && console.log('[FilterPanel] isModalOpen is true, attempting to render AddEditFilterModal.')}
      {isModalOpen && (
        <AddEditFilterModal
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          onSave={handleSaveFilter}
          existingFilter={editingFilter}
          filterDefinitions={filterDefinitions} // Use passed definitions
          // Pass FilterPanel's own styles to the modal, so it can pass them to dynamic config components
          configComponentStyles={styles} 
        />
      )}
    </div>
  );
}

export default FilterPanel; 