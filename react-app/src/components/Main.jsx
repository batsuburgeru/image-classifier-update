import Hero from "./Hero.jsx";
import "../css/Main.css";
import React, { useState, useRef, useEffect } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faArrowLeftLong,
  faArrowRightLong,
} from "@fortawesome/free-solid-svg-icons";

function Main() {
  /* -- Carousel Navigation -- */
  const [activeIndex, setActiveIndex] = useState(0);
  const slidesRef = useRef(null);

  useEffect(() => {
    slidesRef.current = document.getElementsByTagName("article");
  }, []);

  const handleLeftClick = () => {
    const nextIndex =
      activeIndex - 1 >= 0 ? activeIndex - 1 : slidesRef.current.length - 1;

    const currentSlide = document.querySelector(
        `[data-index="${activeIndex}"]`
      ),
      nextSlide = document.querySelector(`[data-index="${nextIndex}"]`);

    currentSlide.dataset.status = "after";

    nextSlide.dataset.status = "becoming-active-from-before";

    setTimeout(() => {
      nextSlide.dataset.status = "active";
      setActiveIndex(nextIndex);
    });
  };

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

  /* -- Mobile Nav Toggle -- */

  const nav = document.querySelector("nav");

  const handleNavToggle = () => {
    nav.dataset.transitionable = "true";

    nav.dataset.toggled = nav.dataset.toggled === "true" ? "false" : "true";
  };

  window.matchMedia("(max-width: 800px)").onchange = (e) => {
    nav.dataset.transitionable = "false";

    nav.dataset.toggled = "false";
  };

  const inputFileRef = useRef(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [imageName, setImageName] = useState(null);

  const handleSelectImageClick = () => {
    inputFileRef.current.click();
  };

  const handleFileChange = (event) => {
    const image = event.target.files[0];
    if (image.size < 5000000) {
      const reader = new FileReader();
      reader.onload = () => {
        setImageSrc(reader.result);
        setImageName(image.name);
      };
      reader.readAsDataURL(image);
    } else {
      alert("Image size more than 2MB");
    }
  };
  return (
    <>
      <main>
        <article data-index="0" data-status="active">
          <div className="article-hero article-section">
            <Hero></Hero>
          </div>
          <div className="article-description-section article-section">
            <p>
              This is a website dedicated to image classification. Determine
              whether a picture contains Buffalo, Bison or, Moose. Click the
              Right Arrow to proceed.
            </p>
          </div>
          <div className="article-title-section article-section">
            <h2>Image Classifier</h2>
          </div>
          <div className="article-nav-section article-section">
            <button
              className="article-nav-button"
              type="button"
              onClick={handleLeftClick}
            >
              <FontAwesomeIcon icon={faArrowLeftLong} />
            </button>
            <button
              className="article-nav-button"
              type="button"
              onClick={handleRightClick}
            >
              <FontAwesomeIcon icon={faArrowRightLong} />
            </button>
          </div>
        </article>
        <article data-index="1" data-status="inactive">
          <div class="article-hero article-section">
            <div className="container">
              <input
                type="file"
                id="file"
                accept="image/*"
                ref={inputFileRef}
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
              <div className="img-area" data-img={imageName}>
                <i className="bx bxs-cloud-upload icon"></i>
                <h3>Upload Image</h3>
                <p>
                  Image size must be less than <span>5MB</span>
                </p>
                {imageSrc && <img src={imageSrc} alt={imageName} />}
              </div>
              <button className="select-image" onClick={handleSelectImageClick}>
                Select Image
              </button>
            </div>
          </div>
          <div class="article-description-section article-section">
            <p>Result</p>
          </div>
          <div class="article-title-section article-section">
            <h2>Input Image to Begin</h2>
            <i class="fa-light fa-plus-large"></i>
          </div>
          <div class="article-nav-section article-section">
            <button
              class="article-nav-button"
              type="button"
              onClick={handleLeftClick}
            >
              <FontAwesomeIcon icon={faArrowLeftLong} />
            </button>
            <button
              class="article-nav-button"
              type="button"
              onClick={handleRightClick}
            >
              <FontAwesomeIcon icon={faArrowRightLong} />
            </button>
          </div>
        </article>
      </main>
      ;
    </>
  );
}

export default Main;
