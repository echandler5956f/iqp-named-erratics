# Design System Documentation

## Overview

This design system provides a comprehensive foundation for the Glacial Erratics Map application, ensuring consistent, professional, and accessible UI components throughout the project.

## Architecture

```
src/styles/
├── design-tokens.css    # Core design tokens (colors, spacing, typography)
├── global.css          # Global styles and base component classes
└── README.md           # This documentation
```

## Design Tokens

All design values are defined as CSS custom properties in `design-tokens.css`. Always use these tokens instead of hardcoded values.

### Colors

```css
/* Primary Colors */
var(--color-primary-500)    /* Main brand color */
var(--color-primary-600)    /* Hover/active states */

/* Neutral Colors */
var(--color-neutral-0)      /* Pure white */
var(--color-neutral-50)     /* Very light gray */
var(--color-neutral-500)    /* Medium gray */
var(--color-neutral-900)    /* Dark text */

/* Semantic Colors */
var(--color-success-500)    /* Success messages */
var(--color-warning-500)    /* Warning messages */
var(--color-error-500)      /* Error messages */
```

### Typography

```css
/* Font Sizes */
var(--font-size-xs)         /* 12px */
var(--font-size-sm)         /* 14px */
var(--font-size-base)       /* 16px */
var(--font-size-lg)         /* 18px */
var(--font-size-xl)         /* 20px */

/* Font Weights */
var(--font-weight-normal)   /* 400 */
var(--font-weight-medium)   /* 500 */
var(--font-weight-semibold) /* 600 */
var(--font-weight-bold)     /* 700 */
```

### Spacing

Based on an 8px grid system:

```css
var(--spacing-1)            /* 4px */
var(--spacing-2)            /* 8px */
var(--spacing-3)            /* 12px */
var(--spacing-4)            /* 16px */
var(--spacing-6)            /* 24px */
var(--spacing-8)            /* 32px */
```

## Component Classes

### Buttons

Use the `.btn` base class with modifiers:

```html
<!-- Primary button -->
<button class="btn btn--primary">Save</button>

<!-- Secondary button -->
<button class="btn btn--secondary">Cancel</button>

<!-- Small button -->
<button class="btn btn--primary btn--sm">Small</button>

<!-- Large button -->
<button class="btn btn--primary btn--lg">Large</button>
```

Or use the React Button component:

```jsx
import { Button } from '../components/ui';

<Button variant="primary" size="lg">Save</Button>
```

### Cards

```html
<div class="card">
  <div class="card__header">
    <h3>Card Title</h3>
  </div>
  <div class="card__body">
    <p>Card content goes here.</p>
  </div>
  <div class="card__footer">
    <button class="btn btn--primary">Action</button>
  </div>
</div>
```

Or use the React Card component:

```jsx
import { Card } from '../components/ui';

<Card hover shadow="lg">
  <Card.Header>
    <h3>Card Title</h3>
  </Card.Header>
  <Card.Body>
    <p>Card content goes here.</p>
  </Card.Body>
  <Card.Footer>
    <Button variant="primary">Action</Button>
  </Card.Footer>
</Card>
```

### Form Elements

Form elements automatically use design tokens:

```html
<div class="form-group">
  <label class="form-label" for="email">Email</label>
  <input type="email" id="email" placeholder="Enter your email">
  <span class="form-error">This field is required</span>
</div>
```

## Utility Classes

### Container

```html
<div class="container">
  <!-- Content will be centered with consistent padding -->
</div>
```

### Screen Reader Only

```html
<span class="sr-only">This text is only for screen readers</span>
```

## Best Practices

### 1. Always Use Design Tokens

❌ **Don't:**
```css
.my-component {
  color: #3b82f6;
  padding: 16px;
  border-radius: 8px;
}
```

✅ **Do:**
```css
.my-component {
  color: var(--color-primary-500);
  padding: var(--spacing-4);
  border-radius: var(--radius-lg);
}
```

### 2. Use BEM Naming Convention

❌ **Don't:**
```css
.filterPanel .addButton {
  /* styles */
}
```

✅ **Do:**
```css
.filter-panel__add-button {
  /* styles */
}
```

### 3. Mobile-First Responsive Design

```css
.component {
  /* Mobile styles first */
  padding: var(--spacing-4);
}

@media (min-width: 768px) {
  .component {
    /* Tablet and desktop styles */
    padding: var(--spacing-6);
  }
}
```

### 4. Use Semantic HTML

❌ **Don't:**
```html
<div class="btn" onclick="submit()">Submit</div>
```

✅ **Do:**
```html
<button type="submit" class="btn btn--primary">Submit</button>
```

## Accessibility

Our design system includes:

- ✅ Focus indicators
- ✅ Proper color contrast ratios
- ✅ Screen reader support
- ✅ Keyboard navigation
- ✅ Reduced motion support

## Migration Guide

### From Old System

1. Replace old CSS imports with our new global styles
2. Update hardcoded colors to use design tokens
3. Replace custom button styles with `.btn` classes
4. Use our Card component for consistent layouts

### Example Migration

**Before:**
```css
.my-button {
  background: #3b82f6;
  padding: 12px 16px;
  border-radius: 6px;
}
```

**After:**
```css
.my-button {
  /* Use existing btn class */
}
```

```html
<!-- Before -->
<button class="my-button">Click me</button>

<!-- After -->
<button class="btn btn--primary">Click me</button>
```

## Adding New Components

When creating new components:

1. Use design tokens for all values
2. Follow BEM naming convention
3. Include hover and focus states
4. Add responsive styles
5. Ensure accessibility compliance
6. Document usage examples

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Resources

- [CSS Custom Properties (MDN)](https://developer.mozilla.org/en-US/docs/Web/CSS/--*)
- [BEM Methodology](https://getbem.com/)
- [WCAG Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/) 