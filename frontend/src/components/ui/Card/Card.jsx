import './Card.css';

function Card({ 
  children, 
  className = '',
  variant = 'default',
  padding = 'default',
  shadow = 'default',
  hover = false,
  ...props 
}) {
  const baseClass = 'card';
  const variantClass = variant !== 'default' ? `card--${variant}` : '';
  const paddingClass = padding !== 'default' ? `card--padding-${padding}` : '';
  const shadowClass = shadow !== 'default' ? `card--shadow-${shadow}` : '';
  const hoverClass = hover ? 'card--hover' : '';
  
  const classes = [
    baseClass,
    variantClass,
    paddingClass,
    shadowClass,
    hoverClass,
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={classes} {...props}>
      {children}
    </div>
  );
}

function CardHeader({ children, className = '', ...props }) {
  return (
    <div className={`card__header ${className}`} {...props}>
      {children}
    </div>
  );
}

function CardBody({ children, className = '', ...props }) {
  return (
    <div className={`card__body ${className}`} {...props}>
      {children}
    </div>
  );
}

function CardFooter({ children, className = '', ...props }) {
  return (
    <div className={`card__footer ${className}`} {...props}>
      {children}
    </div>
  );
}

Card.Header = CardHeader;
Card.Body = CardBody;
Card.Footer = CardFooter;

export default Card; 