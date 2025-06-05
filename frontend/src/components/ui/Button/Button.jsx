import './Button.css';

function Button({ 
  children, 
  variant = 'primary', 
  size = 'base', 
  disabled = false,
  loading = false,
  className = '',
  type = 'button',
  ...props 
}) {
  const baseClass = 'btn';
  const variantClass = `btn--${variant}`;
  const sizeClass = size !== 'base' ? `btn--${size}` : '';
  const disabledClass = disabled || loading ? 'btn--disabled' : '';
  const loadingClass = loading ? 'btn--loading' : '';
  
  const classes = [
    baseClass,
    variantClass,
    sizeClass,
    disabledClass,
    loadingClass,
    className
  ].filter(Boolean).join(' ');

  return (
    <button 
      type={type}
      className={classes}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <span className="btn__spinner" aria-hidden="true"></span>}
      <span className={loading ? 'btn__text--loading' : 'btn__text'}>
        {children}
      </span>
    </button>
  );
}

export default Button; 