export default {
  render({ model, el }) {
    el.classList.add("pr-plot-controls");
    // Own the style element per view: closing an old output must not remove
    // styles used by a newly rendered comparison.
    const style = document.createElement("style");
    style.textContent = model.get("stylesheet");
    el.append(style);
    const buttons = [];
    const sliders = [];
    const select = (group, values) => {
      model.set("selected", { ...model.get("selected"), [group.id]: values });
      model.save_changes();
    };
    for (const group of model.get("groups")) {
      const section = document.createElement("section");
      section.dataset.group = group.id;
      section.setAttribute("aria-label", group.label);
      const heading = document.createElement("div");
      heading.className = "pr-plot-heading";
      const label = document.createElement("strong");
      label.textContent = group.label;
      heading.append(label);
      if (group.kind === "slider") {
        const all = document.createElement("button");
        all.type = "button";
        all.textContent = "ALL";
        all.setAttribute("aria-label", `All ${group.label}`);
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = String(group.options.length - 1);
        slider.step = "1";
        slider.value = "0";
        slider.setAttribute("aria-label", group.label);
        const value = document.createElement("output");
        let last = group.options[0];
        slider.oninput = () => {
          last = group.options[Number(slider.value)];
          select(group, [last]);
        };
        all.onclick = () => select(group,
          (model.get("selected")[group.id] || []).includes("ALL") ? [last] : ["ALL"]);
        heading.append(all, value);
        section.append(heading, slider);
        el.append(section);
        sliders.push({ group, slider, all, value, remember: x => { last = x; } });
        continue;
      }
      if (group.multiple) {
        for (const [title, values] of [["All", group.options], ["None", []]]) {
          const action = document.createElement("button");
          action.type = "button";
          action.textContent = title;
          action.setAttribute("aria-label", `${title} ${group.label.toLowerCase()}`);
          action.onclick = () => select(group, values);
          heading.append(action);
        }
      }
      section.append(heading);
      const choices = document.createElement("div");
      choices.className = "pr-plot-choices";
      for (const [index, option] of group.options.entries()) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "pr-legend-item";
        const sample = group.samples?.[index];
        if (sample) {
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
          svg.setAttribute("viewBox", "0 0 42 16");
          svg.setAttribute("width", "42");
          svg.setAttribute("height", "16");
          svg.setAttribute("aria-hidden", "true");
          const line = document.createElementNS(svg.namespaceURI, "line");
          for (const [key, value] of Object.entries({
            x1: "2", x2: "40", y1: "8", y2: "8",
            stroke: sample.color, "stroke-width": "2",
            "stroke-dasharray": sample.dash || "none",
          })) line.setAttribute(key, value);
          svg.append(line);
          button.append(svg);
        }
        const text = document.createElement("span");
        text.textContent = option;
        button.append(text);
        button.onclick = () => {
          const current = model.get("selected")[group.id] || [];
          select(group, group.kind === "toggle"
            ? (current.includes(option) ? [] : [option])
            : group.multiple
            ? (current.includes(option) ? current.filter(x => x !== option) : [...current, option])
            : [option]);
        };
        buttons.push({ button, group: group.id, option });
        choices.append(button);
      }
      section.append(choices);
      el.append(section);
    }
    const update = () => {
      const selected = model.get("selected");
      for (const { button, group, option } of buttons) {
        button.setAttribute("aria-pressed", String((selected[group] || []).includes(option)));
      }
      for (const { group, slider, all, value, remember } of sliders) {
        const current = selected[group.id]?.[0] || group.options[0];
        const aggregate = current === "ALL";
        slider.disabled = aggregate;
        all.setAttribute("aria-pressed", String(aggregate));
        value.textContent = aggregate ? (group.all_label || "All sizes") : current;
        if (!aggregate) {
          slider.value = String(group.options.indexOf(current));
          slider.setAttribute("aria-valuetext", current);
          remember(current);
        }
      }
    };
    model.on("change:selected", update);
    update();
    let chart = null;
    const onChartWheel = event => {
      // Plotly handles zoom on its inner drag layer first. Keep the same
      // wheel event from scrolling the notebook when it bubbles out.
      if (event.target.closest?.(".js-plotly-plot")) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    let lastWidth = 0;
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const current = el.closest(".pr-comparison")?.firstElementChild;
        if (!current) return;
        if (current !== chart) {
          if (chart) {
            observer.unobserve(chart);
            chart.removeEventListener("wheel", onChartWheel);
          }
          chart = current;
          chart.addEventListener("wheel", onChartWheel, { passive: false });
          observer.observe(chart);
        }
        const width = chart.clientWidth;
        if (width >= 10 && width !== lastWidth) {
          lastWidth = width;
          model.send({ type: "chart_resize", width });
        }
      });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    measure();
    return () => {
      observer.disconnect();
      if (chart) chart.removeEventListener("wheel", onChartWheel);
      cancelAnimationFrame(frame);
      model.off("change:selected", update);
      el.replaceChildren();
    };
  }
};
