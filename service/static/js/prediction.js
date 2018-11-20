$(document).ready(function() {
    $('.bar').each(function() {
        var perc = +$(this).attr('data-perc');
        $(this).height((300 * perc));
        var perc_text = Math.round((perc * 100) * 100) / 100;
        $(this).closest('.bar-box').find('p').text(perc_text + '%');
    });
});
