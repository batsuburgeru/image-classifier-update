import "../css/Navbar.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBars } from "@fortawesome/free-solid-svg-icons";
import { faGithub } from "@fortawesome/free-brands-svg-icons";
import React, { useState, useRef, useEffect } from "react";

function Navbar() {
  /* -- Carousel Navigation -- */
  const [activeIndex, setActiveIndex] = useState(0);
  const slidesRef = useRef(null);

  useEffect(() => {
    slidesRef.current = document.getElementsByTagName("article");
  }, []);

  const handleRightClick = (event) => {
    event.preventDefault();

    const nextIndex =
      activeIndex + 1 <= slidesRef.current.length - 1 ? activeIndex + 1 : 0;

    const currentSlide = document.querySelector(
        `[data-index="${activeIndex}"]`
      ),
      nextSlide = document.querySelector(`[data-index="${nextIndex}"]`);

    currentSlide.dataset.status = "before";

    nextSlide.dataset.status = "becoming-active-from-after";

    setTimeout(() => {
      nextSlide.dataset.status = "active";
      setActiveIndex(nextIndex);
    });
  };
  return (
    <>
      <nav data-toggled="false" data-transitionable="false">
        <div id="nav-logo-section" className="nav-section">
          <a onClick={handleRightClick}>Image Classification Experiment</a>
        </div>
        <div id="nav-mobile-section">
          <div id="nav-link-section" className="nav-section">
            <a>BATO</a>
            <a>BERNAL</a>
          </div>
          <div id="nav-social-section" className="nav-section">
            <a href="#">
              <FontAwesomeIcon icon={faGithub} />
            </a>
          </div>
          <div id="nav-contact-section" className="nav-section">
            <a onClick={handleRightClick}>IMAGE CLASSIFIER</a>
          </div>
        </div>
        <button
          id="nav-toggle-button"
          type="button"
          onclick="handleNavToggle()"
        >
          <span>Menu</span>
          <FontAwesomeIcon icon={faBars} />
        </button>
      </nav>
    </>
  );
}

export default Navbar;
