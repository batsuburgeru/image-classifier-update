function Hero() {
  const wand = document.getElementById("wand"),
    tiles = document.querySelectorAll(".tile");

  const xy = (x, y) => ({ x, y }),
    px = (value) => `${value}px`,
    deg = (value) => `${value}deg`,
    clamp = (value, min, max) => Math.max(Math.min(value, max), min);

  const updateMouse = (mouseX, mouseY) => {
    const { innerWidth: windowWidth, innerHeight: windowHeight } = window;

    const mouse = {
      position: xy(mouseX, mouseY),
      decimal: xy(mouseX / windowWidth, mouseY / windowHeight),
      multiplier: xy(1, 0.4),
      offset: xy(windowWidth * -0.15, windowHeight * 0.1),
      modifiedPosition: xy(0, 0),
    };

    mouse.modifiedPosition.x =
      mouse.position.x * mouse.multiplier.x + mouse.offset.x;
    mouse.modifiedPosition.y =
      mouse.position.y * mouse.multiplier.y + mouse.offset.y;

    return mouse;
  };

  const revealImages = (mouseX) => {
    for (const tile of tiles) {
      const dimensions = tile.getBoundingClientRect(),
        relativeMouseX = mouseX - dimensions.left,
        mouseXAsDecimal = clamp(relativeMouseX / dimensions.width, 0, 1);

      const opacity = mouseXAsDecimal,
        blur = 1 - mouseXAsDecimal;

      tile.style.setProperty("--opacity", opacity);
      tile.style.setProperty("--blur", blur);
    }
  };

  const getWandStyles = (mouse) => ({
    left: px(mouse.modifiedPosition.x),
    top: px(mouse.modifiedPosition.y),
    rotate: deg(mouse.decimal.x * 20 - 10),
  });

  window.onmousemove = (e) => {
    const mouse = updateMouse(e.clientX, e.clientY),
      wandStyles = getWandStyles(mouse);

    wand.animate(wandStyles, { duration: 400, fill: "forwards" });

    revealImages(mouse.modifiedPosition.x);
  };
  return (
    <>
      <div id="wand">
        <div class="cap"></div>
      </div>{" "}
      <div id="tiles">
        <div class="tile">
          <i class="fa-solid fa-image"></i>
          <img src="https://assets.codepen.io/1468070/edgar-infocus-eCSsB3sIEHs-unsplash.jpg?width=1000&height=1250&format=auto&quality=70" />
        </div>
        <div class="tile">
          <i class="fa-solid fa-image"></i>
          <img src="https://assets.codepen.io/1468070/eugene-golovesov-ItyV8UzZzzw-unsplash.jpg?width=818&height=1228&format=auto&quality=80" />
        </div>
        <div class="tile">
          <i class="fa-solid fa-image"></i>
          <img src="https://assets.codepen.io/1468070/alex-shuper-3OpOV8ntLj4-unsplash.jpg?width=1000&height=1500&format=auto&quality=80" />
        </div>
      </div>
    </>
  );
}

export default Hero;
