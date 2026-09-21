document.addEventListener("DOMContentLoaded", () => {
    const cards = document.querySelectorAll(".card");
    cards.forEach((card) => {
        card.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                card.click();
            }
        });
    });
});
