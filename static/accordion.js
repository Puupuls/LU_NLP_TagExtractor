let cur_open = null;

function toggleSection(element){
    if(element === cur_open) {
        collapseSection(cur_open);
        cur_open = null;
    } else {
        if(cur_open)
            collapseSection(cur_open);
        expandSection(element);
        cur_open = element;
    }
}

$(
    () => {
        $(".accordion_head").on("click", function () {
            toggleSection($(this).parent('.accordion_card').children('.accordion_body')[0]);
        })
    }
)

function collapseSection(element) {
    // get the height of the element's inner content, regardless of its actual size
    let sectionHeight = element.scrollHeight;

    // temporarily disable all css transitions
    let elementTransition = element.style.transition;
    element.style.transition = '';
    $(element).parent().removeClass("open");
    // on the next frame (as soon as the previous style change has taken effect),
    // explicitly set the element's height to its current pixel height, so we
    // aren't transitioning out of 'auto'
    requestAnimationFrame(function() {
        element.style.height = sectionHeight + 'px';
        element.style.transition = elementTransition;

        // on the next frame (as soon as the previous style change has taken effect),
        // have the element transition to height: 0
        requestAnimationFrame(function() {
            element.style.height = 0 + 'px';
        });
    });

    // mark the section as "currently collapsed"
    element.setAttribute('data-collapsed', 'true');
}

function expandSection(element) {
    // get the height of the element's inner content, regardless of its actual size
    let sectionHeight = element.scrollHeight;
    $(element).parent().addClass("open");
    // have the element transition to the height of its inner content
    element.style.height = sectionHeight + 'px';

    // when the next css transition finishes (which should be the one we just triggered)
    element.addEventListener('transitionend', function(e) {
        // remove this event listener so it only gets triggered once
        element.removeEventListener('transitionend', arguments.callee);

        // remove "height" from the element's inline styles, so it can return to its initial value
        element.style.height = 'auto';
    });

    // mark the section as "currently not collapsed"
    element.setAttribute('data-collapsed', 'false');
}