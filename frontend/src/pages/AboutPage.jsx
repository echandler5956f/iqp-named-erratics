import Header from '../components/Layout/Header';
import './AboutPage.css';

function AboutPage() {
  return (
    <div className="about-page">
      <Header />
      <div className="about-content">
        <h1>About Glacial Erratics</h1>
        
        <section className="about-section">
          <h2>What are Glacial Erratics?</h2>
          <p>
            Glacial erratics are rocks that differ from the type of rock native to the area in which they rest. They were carried by glacial ice, often over distances of hundreds of kilometers, before being deposited as the glacier melted.
          </p>
          <p>
            Erratics can range in size from pebbles to large boulders weighing thousands of tons. They provide valuable information about patterns of glacial movement and can be used to reconstruct the paths of prehistoric ice sheets.
          </p>
        </section>
        
        <section className="about-section">
          <h2>Cultural Significance</h2>
          <p>
            Throughout history, many glacial erratics have held cultural or religious significance for local populations. Some were used as landmarks, boundary markers, or ceremonial sites. Many have associated legends or folklore that often predate scientific understanding of their glacial origins.
          </p>
          <p>
            Today, notable erratics are often protected as geological heritage sites and serve as educational resources for understanding Earth's glacial history.
          </p>
        </section>
        
        <section className="about-section">
          <h2>About This Project</h2>
          <p>
            The Glacial Erratics Map is a comprehensive database of named glacial erratics, designed to document and preserve information about these fascinating geological features. Our interactive map allows users to explore erratics worldwide, learn about their characteristics, and understand their geological and cultural significance.
          </p>
          <p>
            This project aims to serve as a resource for geologists, historians, educators, and anyone interested in Earth's glacial history and landscape evolution.
          </p>
        </section>
        
        <section className="about-section">
          <h2>Contact Information</h2>
          <p>
            For questions, corrections, or to contribute information about erratics not currently in our database, please contact the project team at <a href="mailto:info@glacialerratics.org">info@glacialerratics.org</a>.
          </p>
        </section>
      </div>
    </div>
  );
}

export default AboutPage; 